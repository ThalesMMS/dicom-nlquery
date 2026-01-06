import json
import logging
from datetime import date
from typing import List, Dict, Any
from .llm_client import OllamaClient
from .agent_tools import DICOM_TOOLS_SCHEMA, execute_tool

log = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""
Você é um Agente Especialista em Radiologia. Data: {date.today()}.
Sua função é encontrar e recuperar exames médicos complexos.

PROCESSO DE RACIOCÍNIO (ReAct):
1. **BUSCA (Search)**: Use `search_studies` com filtros amplos (sexo, data, modalidade) para achar candidatos.
2. **INSPEÇÃO (Inspect)**: O banco de dados NÃO sabe o que é "Esclerose" ou "Contraste".
   - Você DEVE chamar `inspect_metadata` nos estudos candidatos.
   - LEIA as descrições das séries retornadas (SeriesDescription).
   - Procure termos clínicos como "FLAIR", "T2", "GAD", "+C", "DESMIELINIZANTE".
3. **AÇÃO (Act)**: Somente mova (`move_study`) se você confirmar semanticamente que o exame atende ao pedido.

Não invente UIDs. Use apenas os dados retornados.
"""

class DicomAgent:
    def __init__(self, llm: OllamaClient, dicom_client: Any):
        self.llm = llm
        self.client = dicom_client
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.max_steps = 10

    def run(self, user_query: str):
        self.history.append({"role": "user", "content": user_query})
        
        for step in range(self.max_steps):
            log.info(f"--- Passo {step+1} ---")
            
            # 1. LLM decide o que fazer
            response = self.llm.chat_with_tools(self.history, tools=DICOM_TOOLS_SCHEMA)
            self.history.append(response) # Guarda o "pensamento"

            # 2. Verifica se a LLM quer usar uma ferramenta
            tool_calls = response.get("tool_calls")
            
            if not tool_calls:
                # Se não chamou ferramenta, é a resposta final
                return response.get("content")

            # 3. O Python executa cegamente
            for tool in tool_calls:
                fname = tool["function"]["name"]
                fargs = tool["function"]["arguments"]
                if isinstance(fargs, str):
                    fargs = json.loads(fargs)
                
                log.info(f"🔧 Agente chamando: {fname}({fargs})")
                
                result = execute_tool(fname, fargs, self.client)
                
                # 4. Devolve a "visão" para a LLM
                self.history.append({
                    "role": "tool",
                    "content": result,
                    "name": fname
                })
        
        return "Limite de passos atingido."