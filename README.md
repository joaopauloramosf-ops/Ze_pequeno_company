================================================================================
          PROJETO DE BUSINESS INTELLIGENCE & MODELAGEM DE VENDAS
                     ZÉ PEQUENO PAÇOCAS E ROLHAS LTDA
================================================================================

1. VISÃO GERAL DO PROJETO
--------------------------------------------------------------------------------
Este projeto realiza uma análise exploratória, preditiva e temporal sobre os
dados de vendas da Zé Pequeno Ltda. O objetivo central é fornecer insights 
estratégicos sobre o comportamento das vendas dos produtos, por região, saturação 
de mercado e a transformação digital do canal de vendas (E-Commerce vs. Loja Física).


2. ESTRUTURA E ANÁLISES REALIZADAS
--------------------------------------------------------------------------------
O projeto está dividido em 3 principais análises:

• Análise 1: Validação de Hipótese por Região (Testes Estatísticos)
  - Análise comparativa da quantidade de venda de Rolhas e Paçocas ao longo
    do tempo.

• Análise 2: Modelagem Preditiva de Curva de Crescimento (von Bertalanffy)
  - Análise comparativa do volume e receita de vendas entre diferentes regiões,
    aplicando o modelo não-linear de von Bertalanffy para prever o limite de 
    vendas (L_infinity), taxa de crescimento (k) e o tempo de latência/rampa 
    (t0 - Offset) nas 4 principais regiões.

• Análise 3: Tendência Temporal dos Canais de Venda (Top 4 Regiões)
  - Análise da participação percentual (%) entre E-Commerce e Loja Física.
  - Visualização em Facet Wrap (2x2) com linhas de tendência (regressão linear)
    para acompanhar a transição digital da empresa ao longo do tempo.


3. TECNOLOGIAS E BIBLIOTECAS UTILIZADAS
--------------------------------------------------------------------------------
• Linguagem: Python 3.14
• Manipulação de Dados: Pandas, NumPy
• Modelagem e Ajuste de Curvas: SciPy (curve_fit)
• Visualização de Dados: Matplotlib, Seaborn
• Execução e Testes: Google Colab / VS Code


4. COMO EXECUTAR O PROJETO
--------------------------------------------------------------------------------
1. Clone o repositório:
   git clone https://github.com/joaopauloramosf-ops/Ze_pequeno_company.git

2. Instale as dependências requeridas:
   pip install pandas numpy matplotlib seaborn scipy openpyxl

3. Garanta que o arquivo de dados esteja no diretório do projeto:
   - tabela_vendas_ZePequeno_LIMPA.csv

4. Execute os scripts Python (.py) 




================================================================================
Desenvolvido para o curso SCTEC / Modelagem de Dados
================================================================================
