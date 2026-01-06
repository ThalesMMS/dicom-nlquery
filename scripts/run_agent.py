#!/usr/bin/env python3
import argparse
import logging
import sys
from dicom_nlquery.config import load_config
from dicom_nlquery.dicom_client import DicomClient  # Local client with query_studies
from dicom_nlquery.agent import DicomAgent
from dicom_nlquery.llm_client import OllamaClient

# Logs simples para ver o agente "pensando"
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Ex: 'Mulheres jovens com esclerose (FLAIR)'")
    parser.add_argument("--dest", required=True, help="Node de destino (ex: RADIANT)")
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
    agent = DicomAgent(llm, client)

    # 4. Injeta o destino no contexto
    query_full = f"{args.query}. Se encontrar, mova para '{args.dest}'."
    
    print(f"🚀 Iniciando investigação para: {args.query}")
    resposta = agent.run(query_full)
    
    print(f"\n🤖 Resposta:\n{resposta}")

if __name__ == "__main__":
    main()