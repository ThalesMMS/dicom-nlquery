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
5. **UMA FERRAMENTA POR VEZ**: Em cada turno faça no máximo 1 tool_call. Não chame `inspect_metadata` ou `move_study` junto com `search_studies`; espere o resultado anterior.
6. **UID REAL**: Nunca use placeholders (<...>) ou UIDs inventados. `study_instance_uid` deve vir literalmente de um resultado anterior (apenas dígitos e pontos).
7. **SEXO**: Não inferir sexo por gênero gramatical ("um/uma"). Só use `patient_sex` se o usuário declarar explicitamente.
8. **RESSONÂNCIA**: Se o usuário disser RM/ressonância/MRI, mantenha `modality=MR`. Não troque para US ou outra modalidade.
9. **FETO**: Ao buscar feto/gestante, prefira `study_description="*fet*"` para cobrir “feto/fetal”.

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

    def _search_signature(self, result_str: str) -> str:
        if not result_str:
            return "EMPTY"
        try:
            data = json.loads(result_str)
        except json.JSONDecodeError:
            return result_str.strip()
        if not isinstance(data, list):
            return result_str.strip()
        uids = []
        for item in data:
            if isinstance(item, dict):
                uid = item.get("UID")
                if uid:
                    uids.append(str(uid).strip())
        if not uids:
            return "EMPTY"
        return "|".join(sorted(uids))

    def run(self, user_query: str):
        self.history.append({"role": "user", "content": user_query})
        search_signature = "NO_SEARCH"
        moved_uids_for_search: set[str] = set()

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

            # 3. Executa apenas a primeira ferramenta sugerida; evita sequências sem feedback do resultado
            tools_to_run = tool_calls if isinstance(tool_calls, list) else [tool_calls]
            if len(tools_to_run) > 1:
                log.warning(f"⚠️  LLM retornou {len(tools_to_run)} tool_calls; executando apenas a primeira.")
            tool = tools_to_run[0]
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
            if fname == "move_study":
                uid = str(fargs.get("study_instance_uid", "")).strip()
                dest = str(fargs.get("destination_node", "")).strip()
                if uid and dest:
                    if uid in moved_uids_for_search:
                        result_str = (
                            "SKIP: UID ja movido para os resultados atuais. "
                            "Execute nova busca para tentar novamente."
                        )
                        log.info(f"↩️  Ignorando C-MOVE repetido para UID {uid}.")
                        self.history.append({
                            "role": "tool",
                            "content": result_str,
                            "name": fname
                        })
                        self.history.append({
                            "role": "user",
                            "content": "SISTEMA: Nao repita move_study sem nova busca."
                        })
                        continue
                    moved_uids_for_search.add(uid)

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

            if fname == "search_studies":
                new_signature = self._search_signature(result_str)
                if new_signature != search_signature:
                    search_signature = new_signature
                    moved_uids_for_search = set()

            # 6. Dica automática se a busca falhar (Evita loop de desculpas)
            if fname == "search_studies" and ("[]" in result_str or "Nenhum resultado" in result_str):
                log.info("💡 Injetando dica para ampliar busca...")
                self.history.append({
                    "role": "user", 
                    "content": "SISTEMA: A busca retornou vazia. Tente novamente removendo filtros restritivos (como descrição, sexo ou data)."
                })

        return "Limite de passos atingido."
