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
3. **SEM ALUCINAÇÃO**: Se `search_studies` retornar vazio, NÃO invente um UID. Faça NOVA busca com menos filtros.
4. **MODALIDADES**: RM->'MR', TC->'CT', RX->'CR'/'DX', US->'US'. "Qualquer exame" -> não preencha modality.
5. **UMA FERRAMENTA POR VEZ**: Em cada turno faça no máximo 1 tool_call.
6. **UID REAL**: Nunca use placeholders (<...>) ou UIDs inventados.
7. **SEXO**: Não inferir sexo por gênero gramatical. Só use `patient_sex` se explícito.
8. **RESSONÂNCIA**: Mantenha `modality=MR` para RM/ressonância/MRI.

FILTRO DE SÉRIE (IMPORTANTE):
- Use `series_description` para filtrar por características da SÉRIE (não do estudo).
- Exemplos: contraste (*gad*, *contrast*, *pos*), sequências (*T1*, *T2*, *DWI*, *FLAIR*).
- O filtro de série é mais preciso que study_description para características técnicas.

FLUXO OBRIGATÓRIO:
1. **Search** com filtros específicos (incluindo series_description se relevante)
2. **Se vazio**: Amplie removendo filtros, MAS guarde mentalmente o critério original
3. **Se ampliou a busca**: OBRIGATÓRIO usar `inspect_metadata` para verificar se o estudo atende ao pedido original
4. **Analise as séries**: Verifique se alguma série corresponde ao critério (ex: "contraste" -> busque séries com GAD/CONTRAST/POS)
5. **Move APENAS se confirmado**: Se nenhum estudo atende ao critério, responda que não encontrou. NÃO mova um estudo aleatório.

EXEMPLO - "RM com contraste":
- Busca: search_studies(modality='MR', series_description='*gad*') ou '*contrast*'
- Se vazio, amplia para só MR
- ANTES de mover: inspect_metadata para ver as séries
- Se não achar série com contraste: "Não encontrei RM com contraste no sistema."
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

    def _inject_analysis_reminder(self, inspect_result: str):
        """Injeta lembrete para analisar séries antes de mover."""
        self.history.append({
            "role": "user",
            "content": (
                "SISTEMA: Analise as séries acima. O estudo ATENDE ao critério original do usuário? "
                "Se NÃO encontrar séries compatíveis (ex: contraste -> GAD/CONTRAST/POS), "
                "NÃO mova este estudo. Responda que não encontrou estudo compatível. "
                "Só use move_study se CONFIRMAR que o estudo atende ao pedido."
            )
        })

    def run(self, user_query: str):
        self.history.append({"role": "user", "content": user_query})
        search_signature = "NO_SEARCH"
        moved_uids_for_search: set[str] = set()
        search_was_broadened = False  # Rastreia se a busca foi ampliada
        inspected_uids: set[str] = set()  # UIDs que foram inspecionados

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
                    # Bloqueia move sem inspect quando busca foi ampliada
                    if search_was_broadened and uid not in inspected_uids:
                        result_str = (
                            "BLOQUEADO: A busca foi ampliada (filtros removidos). "
                            "Você DEVE usar inspect_metadata neste UID antes de mover, "
                            "para confirmar que o estudo atende ao critério original do usuário."
                        )
                        log.warning(f"🛑 Bloqueando move_study sem inspect para UID {uid}")
                        self.history.append({
                            "role": "tool",
                            "content": result_str,
                            "name": fname
                        })
                        continue
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

            # Rastreia UIDs inspecionados
            if fname == "inspect_metadata":
                uid = str(fargs.get("study_instance_uid", "")).strip()
                if uid:
                    inspected_uids.add(uid)

            result = execute_tool(fname, fargs, self.client)
            
            # Após inspect em busca ampliada, lembra de analisar antes de mover
            if fname == "inspect_metadata" and search_was_broadened:
                # Injeta lembrete ANTES de adicionar o resultado ao histórico
                self._inject_analysis_reminder(result)
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
                search_was_broadened = True  # Marca que a próxima busca será ampliada
                self.history.append({
                    "role": "user", 
                    "content": "SISTEMA: A busca retornou vazia. Tente novamente removendo filtros restritivos (como descrição, sexo ou data)."
                })
            
            # 7. Se busca retornou resultados após ser ampliada, exige inspect
            if fname == "search_studies" and search_was_broadened and "UID" in result_str:
                self.history.append({
                    "role": "user",
                    "content": "SISTEMA: Busca ampliada retornou resultados. OBRIGATÓRIO: Use inspect_metadata no UID mais provável ANTES de mover, para confirmar que atende ao critério original."
                })

        return "Limite de passos atingido."
