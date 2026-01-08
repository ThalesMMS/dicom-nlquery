#!/usr/bin/env python3
import argparse
import logging
import sys

import anyio

from dicom_nlquery.agent import DicomAgent
from dicom_nlquery.config import load_config
from dicom_nlquery.dicom_client import DicomClient  # Local client with query_studies
from dicom_nlquery.llm_client import OllamaClient
from dicom_nlquery.lexicon import load_lexicon
from dicom_nlquery.mcp_client import McpSession, build_stdio_server_params
from dicom_nlquery.models import AgentPhase
from dicom_nlquery.node_registry import NodeRegistry
from dicom_nlquery.resolver import resolve_request

# Logs simples para ver o agente "pensando"
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query",
        help="Ex: 'RM de crânio de mulheres de 20 a 40 anos para RADIANT'",
    )
    args = parser.parse_args()

    # 1. Config
    config = load_config("config.yaml")
    
    # 2. DICOM Client (In-Process para performance)
    # Assume que o dicom-mcp está configurado
    if not config.mcp:
         print("Erro: Configure a seção mcp no config.yaml")
         return

    # Carrega a config do dicom-mcp (gambiarra útil para pegar host/port)
    # Em produção, você carregaria o YAML do dicom-mcp corretamente
    client = DicomClient(
        host="localhost", 
        port=4242, 
        calling_aet="MCPSCU", 
        called_aet="ORTHANC"
    )

    # 3. Inicia Agente
    llm = OllamaClient.from_config(config.llm)
    lexicon = None
    if config.lexicon is not None:
        lexicon = load_lexicon(config.lexicon.path, config.lexicon.synonyms)
    resolver = None
    confirmation_config = None
    require_confirmation = True
    if config.resolver and config.resolver.enabled:
        if not config.mcp:
            print("Erro: Configure a seção mcp no config.yaml")
            return

        async def _fetch_nodes():
            server_params = build_stdio_server_params(config.mcp)
            async with McpSession(server_params, config.mcp) as session:
                return await session.list_dicom_nodes()

        try:
            payload = anyio.run(_fetch_nodes)
        except Exception as exc:
            print(f"Erro ao carregar list_dicom_nodes: {exc}")
            return
        nodes = payload.get("nodes") if isinstance(payload, dict) else None
        if not isinstance(nodes, list):
            print("Erro: list_dicom_nodes retornou formato invalido")
            return
        registry = NodeRegistry.from_tool_payload(nodes)
        resolver = lambda q: resolve_request(q, registry, llm)
        confirmation_config = config.resolver.confirmation
        require_confirmation = config.resolver.require_confirmation

    agent = DicomAgent(
        llm,
        client,
        lexicon=lexicon,
        resolver=resolver,
        confirmation_config=confirmation_config,
        require_confirmation=require_confirmation,
    )

    print(f"🚀 Iniciando investigação para: {args.query}")
    resposta = agent.run(args.query)
    while agent.state.phase in {AgentPhase.CONFIRM, AgentPhase.RESOLVE}:
        print(f"\n🤖 Resposta:\n{resposta}")
        try:
            followup = input("> ")
        except EOFError:
            return
        resposta = agent.run(followup)

    print(f"\n🤖 Resposta:\n{resposta}")

if __name__ == "__main__":
    main()
