from typing import Dict, Any, List
from app.services.document_processor import DocumentProcessor
from app.services.vector_service import VectorService
from app.models.document import Document
import logging

logger = logging.getLogger(__name__)

class AdminController:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_service = VectorService()
        
    async def load_documents_from_directory(
        self, 
        directory_path: str,
        validate_directory: bool = True
    ) -> Dict[str, Any]:
        """
        LÓGICA DE NEGÓCIO: Carrega documentos de um diretório para o sistema RAG
        
        PROCESSO COMPLETO:
        1. Validações de negócio
        2. Carregamento e processamento de documentos
        3. Divisão em chunks
        4. Indexação no vector store
        5. Cálculo de métricas
        6. Logging de resultados
        
        Args:
            directory_path: Caminho para diretório com arquivos .txt
            validate_directory: Se deve validar se diretório existe
            
        Returns:
            Dict com métricas do carregamento realizado
            
        Raises:
            AdminBusinessException: Para erros de negócio específicos
            Exception: Para erros técnicos inesperados
        """
        try:
            if validate_directory:
                await self._validate_directory_path(directory_path)
            
            documents = self.document_processor.load_documents_from_directory(directory_path)
            
            if not documents:
                logger.warning(f"Nenhum documento encontrado em: {directory_path}")
                return {
                    "success": False,
                    "message": f"Nenhum arquivo .txt encontrado em '{directory_path}'",
                    "total_files": 0,
                    "total_chunks": 0,
                    "processing_details": []
                }
            
            processing_results = await self._process_documents_batch(documents)
        
            total_chunks = sum(result["chunks_created"] for result in processing_results)
            successful_files = sum(1 for result in processing_results if result["success"])
            failed_files = len(processing_results) - successful_files
            
            result = {
                "success": True,
                "message": f"Carregamento concluído: {successful_files} arquivos processados com sucesso",
                "total_files": len(documents),
                "successful_files": successful_files,
                "failed_files": failed_files,
                "total_chunks": total_chunks,
                "average_chunks_per_file": round(total_chunks / len(documents), 2),
                "processing_details": processing_results,
                "directory_processed": directory_path
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erro durante carregamento de documentos: {str(e)}")
            return {
                "success": False,
                "message": f"Erro durante carregamento: {str(e)}",
                "total_files": 0,
                "total_chunks": 0,
                "processing_details": [],
                "error": str(e)
            }
    
    async def _validate_directory_path(self, directory_path: str) -> None:
        import os
        
        if not os.path.exists(directory_path):
            raise AdminBusinessException(
                f"Diretório não encontrado: {directory_path}",
                error_code="DIRECTORY_NOT_FOUND"
            )
        
        if not os.path.isdir(directory_path):
            raise AdminBusinessException(
                f"Caminho não é um diretório: {directory_path}",
                error_code="INVALID_DIRECTORY"
            )
        
        txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        if not txt_files:
            raise AdminBusinessException(
                f"Nenhum arquivo .txt encontrado em: {directory_path}",
                error_code="NO_TXT_FILES"
            )
        
    async def _process_documents_batch(self, documents: List[Document]) -> List[Dict[str, Any]]:
        processing_results = []
        
        for i, document in enumerate(documents, 1):
            logger.info(f"Processando documento {i}/{len(documents)}: {document.title}")
            
            try:
                chunked_docs = self.document_processor.chunk_document(document)
                # Indexar cada chunk no vector store
                chunks_indexed = 0
                for chunk in chunked_docs:
                    chunk_id = await self.vector_service.add_document(chunk)
                    if chunk_id:
                        chunks_indexed += 1
                
                result = {
                    "document_title": document.title,
                    "document_category": document.category,
                    "success": True,
                    "chunks_created": len(chunked_docs),
                    "chunks_indexed": chunks_indexed,
                    "file_source": document.metadata.get("source_file", "unknown")
                }
                
                
            except Exception as e:
                logger.error(f"Erro processando {document.title}: {str(e)}")
                result = {
                    "document_title": document.title,
                    "document_category": document.category or "unknown",
                    "success": False,
                    "chunks_created": 0,
                    "chunks_indexed": 0,
                    "error": str(e),
                    "file_source": document.metadata.get("source_file", "unknown")
                }
            
            processing_results.append(result)
        
        return processing_results
    

class AdminBusinessException(Exception):
    """Exceção para erros de lógica de negócio em operações administrativas"""
    
    def __init__(self, message: str, error_code: str = "ADMIN_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)