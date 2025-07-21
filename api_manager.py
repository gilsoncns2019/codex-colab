import random
import time
import google.generativeai as genai
from typing import List, Dict, Optional

class GeminiAPIManager:
    """
    Gerenciador de múltiplas chaves da API Gemini para redundância e balanceamento de carga.
    """
    
    def __init__(self, api_keys: List[str]):
        """
        Inicializa o gerenciador com uma lista de chaves API.
        
        Args:
            api_keys: Lista de chaves da API Gemini
        """
        if not api_keys:
            raise ValueError("Pelo menos uma chave API deve ser fornecida")
        
        self.api_keys = api_keys
        self.current_key_index = 0
        self.failed_keys = set()  # Conjunto de chaves que falharam
        self.retry_count = {}  # Contador de tentativas por chave
        self.max_retries = 3
        self.retry_delay = 1  # segundos
        
    def get_next_key(self) -> Optional[str]:
        """
        Retorna a próxima chave API disponível usando rotação circular.
        
        Returns:
            Chave API ou None se todas as chaves falharam
        """
        available_keys = [key for key in self.api_keys if key not in self.failed_keys]
        
        if not available_keys:
            # Se todas as chaves falharam, resetar e tentar novamente
            self.failed_keys.clear()
            self.retry_count.clear()
            available_keys = self.api_keys
            
        if not available_keys:
            return None
            
        # Rotação circular entre as chaves disponíveis
        key = available_keys[self.current_key_index % len(available_keys)]
        self.current_key_index = (self.current_key_index + 1) % len(available_keys)
        
        return key
    
    def mark_key_failed(self, api_key: str):
        """
        Marca uma chave como falhada.
        
        Args:
            api_key: Chave que falhou
        """
        self.retry_count[api_key] = self.retry_count.get(api_key, 0) + 1
        
        if self.retry_count[api_key] >= self.max_retries:
            self.failed_keys.add(api_key)
            print(f"⚠️ Chave API marcada como falhada após {self.max_retries} tentativas")
    
    def configure_genai(self, api_key: str):
        """
        Configura o google.generativeai com a chave fornecida.
        
        Args:
            api_key: Chave da API para configurar
        """
        genai.configure(api_key=api_key)
    
    def create_model(self, model_name: str = 'gemini-pro') -> Optional[genai.GenerativeModel]:
        """
        Cria um modelo Gemini com uma chave API válida.
        
        Args:
            model_name: Nome do modelo Gemini
            
        Returns:
            Modelo configurado ou None se todas as chaves falharam
        """
        for attempt in range(len(self.api_keys) + 1):  # +1 para dar uma chance extra
            api_key = self.get_next_key()
            
            if not api_key:
                print("❌ Todas as chaves API falharam")
                return None
            
            try:
                self.configure_genai(api_key)
                model = genai.GenerativeModel(model_name)
                
                # Teste rápido para verificar se a chave funciona
                test_response = model.generate_content("Teste")
                
                print(f"✅ Chave API configurada com sucesso (modelo: {model_name})")
                return model
                
            except Exception as e:
                print(f"⚠️ Falha na chave API: {str(e)}")
                self.mark_key_failed(api_key)
                
                if attempt < len(self.api_keys):
                    print(f"🔄 Tentando próxima chave... (tentativa {attempt + 1})")
                    time.sleep(self.retry_delay)
                
        return None
    
    def generate_content_with_fallback(self, prompt: str, model_name: str = 'gemini-pro', max_attempts: int = None) -> Optional[str]:
        """
        Gera conteúdo com fallback automático entre chaves.
        
        Args:
            prompt: Prompt para o modelo
            model_name: Nome do modelo Gemini
            max_attempts: Número máximo de tentativas (padrão: número de chaves)
            
        Returns:
            Resposta gerada ou None se todas as tentativas falharam
        """
        if max_attempts is None:
            max_attempts = len(self.api_keys)
        
        for attempt in range(max_attempts):
            api_key = self.get_next_key()
            
            if not api_key:
                print("❌ Nenhuma chave API disponível")
                break
            
            try:
                self.configure_genai(api_key)
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                
                print(f"✅ Conteúdo gerado com sucesso (tentativa {attempt + 1})")
                return response.text
                
            except Exception as e:
                print(f"⚠️ Erro na tentativa {attempt + 1}: {str(e)}")
                self.mark_key_failed(api_key)
                
                if attempt < max_attempts - 1:
                    print(f"🔄 Tentando próxima chave...")
                    time.sleep(self.retry_delay)
        
        print("❌ Todas as tentativas falharam")
        return None
    
    def get_status(self) -> Dict:
        """
        Retorna o status atual do gerenciador.
        
        Returns:
            Dicionário com informações de status
        """
        return {
            'total_keys': len(self.api_keys),
            'failed_keys': len(self.failed_keys),
            'available_keys': len(self.api_keys) - len(self.failed_keys),
            'current_key_index': self.current_key_index,
            'retry_counts': self.retry_count.copy()
        }


class EnhancedAgent:
    """
    Classe base para agentes com suporte a múltiplas chaves API.
    """
    
    def __init__(self, name: str, prompt: str, api_manager: GeminiAPIManager, model_name: str = 'gemini-pro'):
        self.name = name
        self.prompt = prompt
        self.api_manager = api_manager
        self.model_name = model_name
        self.model = None
    
    def ensure_model(self):
        """
        Garante que o modelo está configurado e funcionando.
        """
        if not self.model:
            self.model = self.api_manager.create_model(self.model_name)
            
        if not self.model:
            raise RuntimeError(f"Não foi possível configurar o modelo para o agente {self.name}")
    
    def process_with_gemini(self, content_to_process: str) -> str:
        """
        Processa conteúdo usando a API Gemini com fallback automático.
        
        Args:
            content_to_process: Conteúdo a ser processado
            
        Returns:
            Resposta processada
        """
        full_prompt = f"{self.prompt}\n\nConteúdo a ser processado: {content_to_process}"
        
        response = self.api_manager.generate_content_with_fallback(
            prompt=full_prompt,
            model_name=self.model_name
        )
        
        if not response:
            raise RuntimeError(f"Falha ao processar conteúdo no agente {self.name}")
        
        return response
    
    def process(self, content):
        """
        Método abstrato para processamento específico do agente.
        """
        raise NotImplementedError


# Exemplo de uso
if __name__ == "__main__":
    # Exemplo de configuração com múltiplas chaves
    api_keys = [
        "sua_chave_1_aqui",
        "sua_chave_2_aqui", 
        "sua_chave_3_aqui"
    ]
    
    # Criar gerenciador
    api_manager = GeminiAPIManager(api_keys)
    
    # Testar geração de conteúdo
    response = api_manager.generate_content_with_fallback(
        "Explique brevemente o que é inteligência artificial",
        model_name='gemini-pro'
    )
    
    if response:
        print(f"Resposta: {response}")
    
    # Verificar status
    status = api_manager.get_status()
    print(f"Status do gerenciador: {status}")


