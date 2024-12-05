from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
import numpy as np

class PineconeManager:
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        # Create index if it doesn't exist
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=384,  # for all-MiniLM-L6-v2
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=environment
                )
            )
        
        # Connect to the index
        self.index = self.pc.Index(index_name)

    def upsert_document(self, chunks: List[Dict[str, Any]], document_metadata: Dict[str, Any], batch_size: int = 100):
        try:
            print(f"Upserting document with PDF ID: {document_metadata['pdf_id']}")
            vectors = []
            
            for i, chunk in enumerate(chunks):
                vector_id = f"{document_metadata['pdf_id']}_{i}"
                metadata = {
                    'text': chunk['text'],
                    'chunk_id': str(i),
                    'pdf_id': document_metadata['pdf_id'],
                    'file_name': str(document_metadata['filename']),
                    'upload_time': str(document_metadata['upload_time']),
                    'total_pages': str(document_metadata['total_pages']),
                    'file_size': str(document_metadata.get('file_size', 'Unknown')),
                    'author': str(document_metadata.get('author', 'Unknown')),
                    'creator': str(document_metadata.get('creator', 'Unknown')),
                    'producer': str(document_metadata.get('producer', 'Unknown')),
                    'subject': str(document_metadata.get('subject', 'Unknown')),
                    'title': str(document_metadata.get('title', 'Unknown')),
                    'creation_date': str(document_metadata.get('creation_date', 'Unknown')),
                    'page_number': str(chunk.get('page_number', 0))
                }
                vectors.append({
                    'id': vector_id,
                    'values': chunk['embedding'].tolist(),
                    'metadata': metadata
                })

            print(f"Created {len(vectors)} vectors")

            # Delete previous vectors for this PDF if they exist
            try:
                print(f"Deleting previous vectors for PDF ID: {document_metadata['pdf_id']}")
                self.index.delete(filter={'pdf_id': document_metadata['pdf_id']})
            except Exception as e:
                print(f"Error deleting previous vectors: {str(e)}")

            # Batch upsert new vectors
            total_upserted = 0
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                total_upserted += len(batch)
                print(f"Upserted batch of {len(batch)} vectors. Total: {total_upserted}/{len(vectors)}")

            print("Document upsert completed successfully")
            return True
        except Exception as e:
            print(f"Error upserting document: {str(e)}")
            print("Full error:", exc_info=True)
            return False

    def query(self, query_vector: np.ndarray, pdf_id: str = None, top_k: int = 3):
        """Query Pinecone"""
        try:
            print(f"Querying Pinecone with PDF ID: {pdf_id}")
            query_params = {
                'vector': query_vector.tolist(),
                'top_k': top_k,
                'include_metadata': True
            }
            
            if pdf_id:
                query_params['filter'] = {'pdf_id': pdf_id}
                print(f"Added filter: {query_params['filter']}")
            
            print("Query params:", query_params)
            results = self.index.query(**query_params)
            print("Pinecone query results:", results)
            
            if not results.matches:
                print("No matches found in Pinecone")
            else:
                print(f"Found {len(results.matches)} matches")
            
            return results
        except Exception as e:
            print(f"Error querying Pinecone: {str(e)}")
            print("Full error:", exc_info=True)
            return {'matches': []} 