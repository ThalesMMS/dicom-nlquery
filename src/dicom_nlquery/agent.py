import json
import logging
import re
from datetime import date
from typing import List, Dict, Any, Optional
from .llm_client import OllamaClient
from .agent_tools import DICOM_TOOLS_SCHEMA, execute_tool

log = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""
Você é um Agente Especialista em Radiologia e DICOM. Data: {date.today()}.

REGRAS CRÍTICAS DE OPERAÇÃO:
1. **TOOL CALLING**: Use as ferramentas disponíveis. Não simule respostas JSON no texto.
2. **OBSTETRÍCIA/FETAL**: Em exames fetais ("feto", "gestante"), o paciente cadastrado geralmente é a MÃE. **NUNCA** filtre por `patient_sex='M'` para fetos. Use 'F' ou remova o filtro de sexo.
3. **SEM ALUCINAÇÃO**: Se `search_studies` retornar vazio ("[]"), NÃO invente um UID. Sua próxima ação deve ser uma NOVA busca com menos filtros (ex: remover data ou descrição).
4. **MODALIDADES**:
   - RM/Ressonância -> 'MR'
   - TC/Tomografia -> 'CT'
   - RX/Raio-X -> 'CR' ou 'DX'
   - US/Ultrassom -> 'US'
   - "Qualquer exame" -> Não preencha o campo modality.

FLUXO: Search -> (Se vazio: Search Broader) -> (Se achou: Inspect) -> (Se confirmado: Move).
"""

class DicomAgent:
    def __init__(self, llm: OllamaClient, dicom_client: Any):
        self.llm = llm
        self.client = dicom_client
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.max_steps = 10

    def _extract_json_from_text(self, text: str) -> Optional[List[Dict]]:
        """
        Rede de segurança: tenta resgatar chamadas de ferramenta
        quando a LLM escreve o JSON no corpo do texto (alucinação de formato).
        """
        try:
            # Procura por padrões de JSON de ferramenta no texto: {"name": "...", "parameters": {...}}
            match = re.search(r'(\{\s*"name"\s*:\s*".*?"\s*,\s*"(parameters|arguments)"\s*:\s*\{.*?\}\s*\})', text, re.DOTALL)
            if match:
                json_str = match.group(1)
                data = json.loads(json_str)
                
                # Normaliza campos (alguns modelos usam 'arguments' outros 'parameters')
                fname = data.get("name")
                fargs = data.get("parameters") or data.get("arguments") or {}
                
                if fname:
                    log.warning(f"🕵️  Detectado JSON perdido no texto. Convertendo para Tool Call: {fname}")
                    return [{
                        "function": {
                            "name": fname,
                            "arguments": fargs
                        }
                    }]
        except Exception as e:
            pass
        return None

    def run(self, user_query: str):
        self.history.append({"role": "user", "content": user_query})
        
        for step in range(self.max_steps):
            log.info(f"--- Passo {step+1} ---")
            
            # 1. Chama a LLM
            response = self.llm.chat_with_tools(self.history, tools=DICOM_TOOLS_SCHEMA)
            self.history.append(response)

            tool_calls = response.get("tool_calls")
            content = response.get("content", "") or ""

            # 2. Lógica de Fallback (Salva o dia se o JSON vier no texto)
            if not tool_calls and "{" in content:
                rescued = self._extract_json_from_text(content)
                if rescued:
                    tool_calls = rescued

            # Se realmente não tem ferramenta, retorna o texto final
            if not tool_calls:
                return content

            # 3. Chain Breaking: Executa APENAS a primeira ferramenta sugerida
            tool = tool_calls[0]
            fname = tool["function"]["name"]
            fargs = tool["function"]["arguments"]
            
            # Garante que args seja dict
            if isinstance(fargs, str):
                try:
                    fargs = json.loads(fargs)
                except json.JSONDecodeError:
                    pass
            
            log.info(f"🔧 Agente chamando: {fname}({fargs})")
            
            # 4. Execução
            result = execute_tool(fname, fargs, self.client)
            result_str = str(result)
            
            # LOG IMPORTANTE: Ver o que retornou para debug
            preview = (result_str[:150] + '...') if len(result_str) > 150 else result_str
            log.info(f"   ↳ Resultado: {preview}")
            
            # 5. Devolve resultado para a LLM
            self.history.append({
                "role": "tool",
                "content": result_str,
                "name": fname
            })
            
            # 6. Dica automática se a busca falhar (Evita loop de desculpas)
            if fname == "search_studies" and ("[]" in result_str or "Nenhum resultado" in result_str):
                log.info("💡 Injetando dica para ampliar busca...")
                self.history.append({
                    "role": "user", 
                    "content": "SISTEMA: A busca retornou vazia. Tente novamente removendo filtros restritivos (como descrição, sexo ou data)."
                })

            if len(tool_calls) > 1:
                log.info(f"⚠️  Ignorando {len(tool_calls)-1} chamadas subsequentes para manter foco.")

        return "Limite de passos atingido."