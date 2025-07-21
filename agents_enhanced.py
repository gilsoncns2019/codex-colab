import re
from api_manager import GeminiAPIManager, EnhancedAgent

class TranscriberExtractorAgent(EnhancedAgent):
    def __init__(self, api_manager: GeminiAPIManager, model_name: str = 'gemini-pro'):
        prompt = """
        Você é o Agente 1: Transcritor/Extrator de Conteúdo. Sua função é realizar a transcrição fiel de áudios/vídeos ou a extração precisa de texto de PDFs/slides. Você não interpreta, resume ou opina, apenas captura a matéria-prima. Sua saída deve manter a fidelidade ao conteúdo original, incluindo questões de revisão e gabaritos.
        
        Instruções específicas:
        - Mantenha a fidelidade absoluta ao conteúdo original
        - Não remova informações importantes
        - Preserve questões, gabaritos e comentários
        - Identifique e marque claramente bizus e dicas
        """
        super().__init__("Transcritor/Extrator de Conteúdo", prompt, api_manager, model_name)

    def process(self, content):
        """
        Processa o conteúdo bruto extraindo e organizando as informações.
        Combina processamento baseado em regex com IA para maior precisão.
        """
        processed_content = content
        bizus = []
        questions = []

        # Patterns to remove administrative/boilerplate content
        patterns_to_remove = [
            r'(?is)Apresentação do Curso.*?(?=\n\n|\Z)',
            r'(?is)Sumário.*?(?=\n\n|\Z)',
            r'(?is)Índice.*?(?=\n\n|\Z)',
            r'(?is)Conteúdo Introdutório e Administrativo.*?(?=\n\n|\Z)',
            r'(?is)Material Publicitário.*?(?=\n\n|\Z)',
            r'(?is)Metadados de Origem.*?(?=\n\n|\Z)',
            r'(?i)Página \d+',
            r'(?i)Direitos Autorais © \d{4}',
            r'(?is)Seções de Exercícios.*?(?=\n\n|\Z)',
            r'(?is)Exercícios Não Comentados/Resolvidos.*?(?=\n\n|\Z)',
        ]

        for pattern in patterns_to_remove:
            processed_content = re.sub(pattern, 
                                       lambda m: '' if 'Questão' not in m.group(0) and 'Gabarito' not in m.group(0) else m.group(0),
                                       processed_content)

        # Extrair bizus usando regex e IA para validação
        bizu_pattern = r'(?i)(bizu\s*[:!].*?(?:\n|$))'
        bizu_matches = list(re.finditer(bizu_pattern, processed_content))
        
        for match in reversed(bizu_matches):
            bizus.append(match.group(0).strip())
            processed_content = processed_content[:match.start()] + processed_content[match.end():]
        bizus.reverse()

        # Se o conteúdo for complexo, usar IA para identificar bizus adicionais
        if len(processed_content) > 1000:
            try:
                ai_bizus = self.process_with_gemini(f"Identifique dicas, macetes ou bizus importantes no seguinte texto. Retorne apenas as dicas encontradas, uma por linha: {processed_content}")
                if ai_bizus and ai_bizus.strip():
                    for line in ai_bizus.strip().split('\n'):
                        if line.strip() and line.strip() not in [b.strip() for b in bizus]:
                            bizus.append(f"Bizu: {line.strip()}")
            except Exception as e:
                print(f"⚠️ Erro ao processar bizus com IA: {e}")

        # Extrair questões de revisão
        question_pattern = r'(?is)(Questão\s*\d*\s*\(.*?\).*?(?:Alternativa\s*[A-Z]\).*?)*Gabarito\s*[:!].*?Comentário\s*[:!].*?(?:\n\n|\Z))'
        question_matches = list(re.finditer(question_pattern, processed_content))

        for match in reversed(question_matches):
            questions.append(match.group(0).strip())
            processed_content = processed_content[:match.start()] + processed_content[match.end():]
        questions.reverse()

        return {
            'main_content': processed_content.strip(),
            'bizus': bizus,
            'questions': questions
        }

class StructurerVisualizerAgent(EnhancedAgent):
    def __init__(self, api_manager: GeminiAPIManager, model_name: str = 'gemini-pro'):
        prompt = """
        Você é o Agente 2: Estruturador e Visualizador. Sua função é organizar o conteúdo bruto em uma estrutura lógica de Markdown, aplicando heurísticas de visualização para decidir qual diagrama (Mermaid) é mais apropriado para cada tipo de informação.
        
        Instruções específicas:
        - Organize o conteúdo em seções claras e hierárquicas
        - Use formatação Markdown apropriada (títulos, listas, ênfases)
        - Sugira diagramas Mermaid quando apropriado
        - Mantenha a legibilidade e navegabilidade
        - Preserve toda a informação importante
        """
        super().__init__("Estruturador e Visualizador", prompt, api_manager, model_name)

    def process(self, extracted_data):
        main_content = extracted_data.get('main_content', '')
        bizus = extracted_data.get('bizus', [])
        questions = extracted_data.get('questions', [])

        structured_content = "# Conteúdo Estruturado\n\n"
        
        if main_content:
            structured_content += "## Conteúdo Principal\n\n"
            
            # Usar IA para estruturar conteúdo complexo
            if len(main_content) > 500:
                try:
                    ai_structured = self.process_with_gemini(f"""
                    Estruture o seguinte conteúdo em Markdown bem organizado:
                    - Use títulos e subtítulos apropriados (##, ###)
                    - Crie listas quando apropriado
                    - Mantenha parágrafos bem formatados
                    - Preserve toda a informação original
                    - Não adicione informações que não estão no texto original
                    
                    Conteúdo: {main_content}
                    """)
                    structured_content += ai_structured + "\n\n"
                except Exception as e:
                    print(f"⚠️ Erro ao estruturar com IA, usando conteúdo original: {e}")
                    structured_content += main_content + "\n\n"
            else:
                structured_content += main_content + "\n\n"
        
        if bizus:
            structured_content += "## 💡 Bizus e Dicas Importantes\n\n"
            for i, bizu in enumerate(bizus, 1):
                clean_bizu = bizu.replace("Bizu:", "").replace("bizu:", "").strip()
                structured_content += f"> [!TIP] **Bizu {i}:** {clean_bizu}\n\n"
        
        if questions:
            structured_content += "## 📝 Questões de Revisão\n\n"
            for i, question_text in enumerate(questions, 1):
                # Usar IA para melhor formatação das questões
                try:
                    formatted_question = self.process_with_gemini(f"""
                    Formate a seguinte questão em Markdown bem estruturado:
                    - Separe claramente o enunciado, alternativas, gabarito e comentário
                    - Use formatação apropriada para cada seção
                    - Mantenha toda a informação original
                    
                    Questão: {question_text}
                    """)
                    structured_content += f"### Questão {i}\n\n{formatted_question}\n\n---\n\n"
                except Exception as e:
                    print(f"⚠️ Erro ao formatar questão com IA: {e}")
                    # Fallback para formatação manual
                    question_parts_match = re.search(r'(?is)(Questão\s*\d*\s*\(.*?\).*?)(Alternativa\s*[A-Z]\).*?(?:\nAlternativa\s*[A-Z]\).*?)*)(Gabarito\s*[:!].*?)(Comentário\s*[:!].*)', question_text)
                    
                    if question_parts_match:
                        q_body = question_parts_match.group(1).strip()
                        alternatives_raw = question_parts_match.group(2).strip()
                        gabarito = question_parts_match.group(3).strip()
                        comentario = question_parts_match.group(4).strip()

                        structured_content += f"### Questão {i}\n\n"
                        structured_content += f"{q_body}\n\n"
                        
                        formatted_alternatives = []
                        for alt_line in alternatives_raw.split('\n'):
                            if alt_line.strip():
                                formatted_alternatives.append(f"- [ ] {alt_line.strip()}")
                        structured_content += '\n'.join(formatted_alternatives) + "\n\n"
                        
                        structured_content += f"**{gabarito}**\n\n"
                        structured_content += f"**{comentario}**\n\n"
                        structured_content += f"---\n\n"
                    else:
                        structured_content += f"### Questão {i}\n\n{question_text}\n\n---\n\n"

        # Sugerir diagramas usando IA
        try:
            diagram_suggestion = self.process_with_gemini(f"""
            Analise o seguinte conteúdo e determine se seria útil adicionar um diagrama Mermaid.
            Se sim, especifique o tipo (flowchart, sequenceDiagram, graph, mindmap, etc.) e forneça o código Mermaid.
            Se não for apropriado, responda apenas 'Nenhum diagrama necessário'.
            
            Conteúdo: {main_content[:1000]}
            """)
            
            if 'mermaid' in diagram_suggestion.lower() and 'nenhum' not in diagram_suggestion.lower():
                structured_content += "## 📊 Diagrama Ilustrativo\n\n"
                structured_content += diagram_suggestion + "\n\n"
        except Exception as e:
            print(f"⚠️ Erro ao gerar diagrama: {e}")

        return {
            'structured_content': structured_content,
            'has_math': '$' in main_content or 'fórmula' in main_content.lower() or 'equação' in main_content.lower()
        }

class LatexExpertAgent(EnhancedAgent):
    def __init__(self, api_manager: GeminiAPIManager, model_name: str = 'gemini-pro'):
        prompt = """
        Você é o Agente 3: Especialista em LaTeX. Sua função é auxiliar na geração de código LaTeX para fórmulas e expressões matemáticas, garantindo que a sintaxe esteja correta e seja compatível com o MathJax do Obsidian.
        
        Instruções específicas:
        - Valide e corrija sintaxe LaTeX
        - Garanta compatibilidade com MathJax
        - Use $...$ para fórmulas inline
        - Use $$...$$ para fórmulas em bloco
        - Mantenha formatação clara e legível
        """
        super().__init__("Especialista em LaTeX", prompt, api_manager, model_name)

    def process(self, structured_data):
        structured_content = structured_data.get('structured_content', '')
        has_math = structured_data.get('has_math', False)

        if has_math:
            try:
                # Usar IA para validar e melhorar LaTeX
                latex_improved = self.process_with_gemini(f"""
                Revise e melhore a sintaxe LaTeX no seguinte texto:
                - Corrija erros de sintaxe LaTeX
                - Garanta compatibilidade com MathJax do Obsidian
                - Mantenha fórmulas inline com $...$ e em bloco com $$...$$
                - Não altere o conteúdo não-matemático
                - Retorne o texto completo com as correções
                
                Texto: {structured_content}
                """)
                structured_content = latex_improved
            except Exception as e:
                print(f"⚠️ Erro ao processar LaTeX com IA: {e}")
                # Fallback para correção básica com regex
                structured_content = re.sub(r'\$([^$]+?)\$', r'$\1$', structured_content)
                structured_content = re.sub(r'\$\$([^$]+?)\$\$', r'$$\1$$', structured_content)
            
            structured_content += "\n> [!NOTE] 🧮 Este conteúdo contém fórmulas matemáticas renderizadas com LaTeX/MathJax.\n\n"

        return {
            'content': structured_content,
            'latex_processed': has_math
        }

class ObsidianFormatterAgent(EnhancedAgent):
    def __init__(self, api_manager: GeminiAPIManager, model_name: str = 'gemini-pro'):
        prompt = """
        Você é o Agente 4: Formatador Final Obsidian. Sua função é revisar a nota completa para garantir que toda a formatação Markdown esteja correta, que os links internos e externos funcionem, e que a nota seja clara, navegável e otimizada para o uso no Obsidian.
        
        Instruções específicas:
        - Revise toda a formatação Markdown
        - Adicione tags relevantes e úteis
        - Garanta navegabilidade e clareza
        - Otimize para uso no Obsidian
        - Mantenha consistência na formatação
        """
        super().__init__("Formatador Final Obsidian", prompt, api_manager, model_name)

    def process(self, latex_data):
        content = latex_data.get('content', '')
        latex_processed = latex_data.get('latex_processed', False)

        try:
            # Usar IA para revisão final e adição de tags
            final_review = self.process_with_gemini(f"""
            Faça uma revisão final desta nota para Obsidian:
            - Corrija qualquer problema de formatação Markdown
            - Adicione tags relevantes no final (formato: tags: [tag1, tag2, tag3])
            - Garanta que títulos estão bem hierarquizados
            - Verifique se listas e formatações estão corretas
            - Mantenha todo o conteúdo original
            - Retorne apenas o conteúdo revisado, sem comentários adicionais
            
            Nota: {content}
            """)
            
            final_content = final_review
            
        except Exception as e:
            print(f"⚠️ Erro na revisão final com IA: {e}")
            final_content = content
            
            # Adicionar tags básicas se LaTeX foi processado
            if latex_processed and "tags:" not in final_content.lower():
                final_content += "\n---\ntags: [matemática, latex, estudo]\n"

        # Adicionar metadados do Obsidian se não existirem
        if not final_content.startswith("---"):
            metadata = f"---\ncreated: {time.strftime('%Y-%m-%d')}\ntags: [processado, multi-agente]\n---\n\n"
            final_content = metadata + final_content

        return final_content


def process_content_with_enhanced_agents(raw_content, api_keys, model_names=None):
    """
    Processa conteúdo usando agentes aprimorados com múltiplas chaves API.
    
    Args:
        raw_content: Conteúdo bruto a ser processado
        api_keys: Lista de chaves da API Gemini
        model_names: Dicionário com modelos específicos por agente
    
    Returns:
        Conteúdo processado e formatado
    """
    if model_names is None:
        model_names = {
            'TranscriberExtractorAgent': 'gemini-pro',
            'StructurerVisualizerAgent': 'gemini-pro',
            'LatexExpertAgent': 'gemini-pro',
            'ObsidianFormatterAgent': 'gemini-pro'
        }

    # Criar gerenciador de API
    api_manager = GeminiAPIManager(api_keys)
    
    # Criar agentes
    transcriber = TranscriberExtractorAgent(api_manager, model_names.get('TranscriberExtractorAgent'))
    structurer = StructurerVisualizerAgent(api_manager, model_names.get('StructurerVisualizerAgent'))
    latex_expert = LatexExpertAgent(api_manager, model_names.get('LatexExpertAgent'))
    formatter = ObsidianFormatterAgent(api_manager, model_names.get('ObsidianFormatterAgent'))

    print(f"\n🚀 Iniciando Processamento Multi-Agente com Gemini API")
    print(f"📊 Status inicial: {api_manager.get_status()}")

    try:
        # Processamento sequencial pelos agentes
        print(f"\n🔄 Executando Agente 1: {transcriber.name}")
        extracted_data = transcriber.process(raw_content)
        print(f"✅ Agente 1 concluído")

        print(f"\n🔄 Executando Agente 2: {structurer.name}")
        structured_data = structurer.process(extracted_data)
        print(f"✅ Agente 2 concluído")

        print(f"\n🔄 Executando Agente 3: {latex_expert.name}")
        latex_data = latex_expert.process(structured_data)
        print(f"✅ Agente 3 concluído")

        print(f"\n🔄 Executando Agente 4: {formatter.name}")
        final_note = formatter.process(latex_data)
        print(f"✅ Agente 4 concluído")

        print(f"\n🎉 Processamento Multi-Agente Concluído com Sucesso!")
        print(f"📊 Status final: {api_manager.get_status()}")
        
        return final_note

    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")
        print(f"📊 Status no momento do erro: {api_manager.get_status()}")
        raise


if __name__ == "__main__":
    import time
    
    # Exemplo de uso (substitua pelas suas chaves reais)
    test_api_keys = [
        "sua_chave_1_aqui",
        "sua_chave_2_aqui",
        "sua_chave_3_aqui"
    ]

    test_content = """
    Este é um conteúdo de teste com algumas informações importantes. 
    Temos um bizu: Lembre-se da regra de ouro para resolver equações! 
    Uma fórmula matemática importante: $E=mc^2$. 
    Outra fórmula: $$F = ma$$
    E uma questão de revisão: Qual a capital do Brasil? 
    a) São Paulo b) Rio de Janeiro c) Brasília d) Belo Horizonte
    Gabarito: c) Brasília. 
    Comentário: A capital do Brasil é Brasília, fundada em 21 de abril de 1960 e projetada por Oscar Niemeyer.
    """
    
    # Exemplo com modelos específicos por agente
    custom_model_names = {
        'TranscriberExtractorAgent': 'gemini-pro',
        'StructurerVisualizerAgent': 'gemini-pro',
        'LatexExpertAgent': 'gemini-pro',
        'ObsidianFormatterAgent': 'gemini-pro'
    }

    try:
        processed_output = process_content_with_enhanced_agents(
            test_content, 
            test_api_keys, 
            custom_model_names
        )
        print(f"\n📄 SAÍDA FINAL DO SISTEMA:\n{processed_output}")
    except Exception as e:
        print(f"❌ Erro no processamento: {e}")

