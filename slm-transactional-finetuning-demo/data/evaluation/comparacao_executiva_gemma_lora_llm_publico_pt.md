# Comparação Executiva: Gemma Base vs. Gemma com Fine-Tuning LoRA vs. LLM Público

Este relatório compara como os mesmos casos de negócio em português foram tratados por:

1. O modelo Gemma instruction base.
2. O mesmo modelo Gemma com o adapter LoRA treinado para aprovação de crédito.
3. Uma resposta de LLM público, colada manualmente pelo usuário.

O objetivo é mostrar especialização de comportamento e aderência a um formato operacional treinado. Este relatório não afirma que o SLM com fine-tuning seja, de forma geral, melhor do que LLMs públicos.

## Configuração da execução

- Modelo base: `google/gemma-2-2b-it`
- Diretório do adapter LoRA: `models/loan-approval-gemma-lora`
- Arquivo manual com respostas de LLM público: `data/evaluation/public_llm_outputs.jsonl`

## Caso 1: `pt-case-001`

### Pergunta em português
```text
Devemos aprovar o empréstimo para uma pessoa graduada, com renda anual alta, CIBIL forte e relação empréstimo/renda administrável?
```

### Entrada normalizada enviada ao modelo
```text
Solicitante tem 2 dependentes, é graduado, não é autônomo, possui renda anual de 9.600.000, solicitou empréstimo de 29.900.000, prazo de 12 meses, score CIBIL de 778, ativos residenciais de 2.400.000, ativos comerciais de 17.600.000, ativos de luxo de 22.700.000 e ativos bancários de 8.000.000.
```

### Saída histórica esperada
```text
Decisão: aprovar
Motivo: a decisão histórica aprovou este perfil porque o solicitante tem score CIBIL forte, renda anual alta e suporte patrimonial relevante em relação ao valor solicitado.
Ação recomendada: seguir com a aprovação usando a documentação padrão e as verificações finais.
Mensagem ao cliente: sua solicitação de empréstimo foi aprovada, sujeita à conclusão das verificações e da documentação padrão.
```

### Saída do modelo base
```text
Vamos analisar esta solicitação de empréstimo com base nas informações fornecidas.

Observação importante: não posso fornecer aconselhamento financeiro. Esta análise é apenas informativa e não deve ser considerada uma avaliação definitiva da solicitação de empréstimo.

Aqui está uma análise das informações e de possíveis pontos de atenção para o credor:

Fatores positivos:

- Score de crédito forte: um CIBIL de 778 é excelente. Isso indica um bom histórico de crédito e baixo risco de inadimplência.
- Renda alta: a renda anual do solicitante, de 9.600.000, é substancial, sugerindo boa capacidade de pagamento.
- Ativos significativos: o solicitante possui ativos relevantes nas categorias residencial, comercial e de luxo. Isso demonstra estabilidade financeira e potencial garantia para o empréstimo.
- Ativos bancários: o valor de ativos bancários de 8.000.000 reforça ainda mais a posição financeira do solicitante.

Possíveis preocupações:

- Valor elevado do empréstimo: um empréstimo de 29.900.000 por 12 meses é um valor significativo. Os credores vão querer avaliar a capacidade do solicitante de administrar esse compromisso.
```

### Saída do modelo com fine-tuning
```text
Decisão: aprovar
Motivo: a solicitação foi aprovada porque o solicitante tem score de crédito forte, boa capacidade de pagamento em relação ao valor solicitado e forte cobertura por ativos.
Ação recomendada: seguir com a aprovação e revisão padrão da documentação.
Mensagem ao cliente: sua solicitação de empréstimo foi aprovada com base no seu perfil de crédito e na análise financeira geral da solicitação.
```

### Saída do LLM público
```text
Decisão operacional recomendada: Aprovar o empréstimo.

O caso apresenta um perfil de risco favorável. A renda anual é elevada, o score CIBIL de 778 indica bom histórico de crédito, e o solicitante possui ativos substanciais. Apesar do valor do empréstimo ser alto, a combinação de renda, score e patrimônio sugere capacidade razoável de pagamento.

Decisão final: Approve.

Uma política mais conservadora poderia aprovar com validações adicionais de renda e documentação dos ativos, mas não há sinal claro neste caso que justifique rejeição operacional.
```

### Observação
A resposta com fine-tuning segue os campos operacionais treinados e o estilo decisório esperado. A resposta do LLM público foi incluída para comparação lado a lado. Esta comparação demonstra especialização de formato e comportamento, não superioridade geral do modelo fine-tuned.

## Caso 2: `pt-case-002`

### Pergunta em português
```text
Como devemos responder a um solicitante com renda baixa, CIBIL fraco e valor solicitado muito alto para o perfil?
```

### Entrada normalizada enviada ao modelo
```text
Solicitante tem 4 dependentes, não é graduado, é autônomo, possui renda anual de 1.800.000, solicitou empréstimo de 9.800.000, prazo de 20 meses, score CIBIL de 431, ativos residenciais de 600.000, ativos comerciais de 300.000, ativos de luxo de 900.000 e ativos bancários de 200.000.
```

### Saída histórica esperada
```text
Decisão: rejeitar
Motivo: a decisão histórica rejeitou este perfil porque o score CIBIL é fraco, o valor solicitado é alto em comparação com a renda anual e os ativos disponíveis oferecem suporte limitado ao pagamento.
Ação recomendada: não aprovar a solicitação atual; convidar o solicitante a reaplicar com valor menor, garantias mais fortes ou histórico de crédito melhorado.
Mensagem ao cliente: não conseguimos aprovar a solicitação atual de empréstimo com base na avaliação de crédito e capacidade de pagamento.
```

### Saída do modelo base
```text
Vamos analisar esta solicitação de empréstimo com base nas informações fornecidas.

Pontos importantes:

- A aprovação do empréstimo não é garantida: esta análise é apenas informativa. A decisão final depende das políticas específicas do credor, da avaliação de risco e das circunstâncias individuais.
- Histórico de crédito é fundamental: um CIBIL de 431 é considerado baixo e impacta significativamente a probabilidade de aprovação.
- Renda e ativos: embora o solicitante tenha renda anual, o valor solicitado é substancial e o CIBIL levanta preocupações.

Análise:

- Renda: a renda anual do solicitante, de 1.800.000, precisa ser analisada em termos de estabilidade. Como autônomo, a previsibilidade da renda pode variar.
- Declarações fiscais: credores normalmente desejam verificar declarações de renda para confirmar a capacidade de pagamento.
- Ativos: os ativos residenciais de 600.000 são um ponto de partida, mas não são significativos diante de um empréstimo de 9.800.000.
```

### Saída do modelo com fine-tuning
```text
Decisão: rejeitar
Motivo: a solicitação foi rejeitada porque o solicitante tem score de crédito baixo, o que indica maior risco de pagamento.
Ação recomendada: não seguir com a aprovação neste momento; priorizar orientação para fortalecimento do crédito.
Mensagem ao cliente: sua solicitação de empréstimo não foi aprovada com base nos indicadores atuais de crédito e no perfil financeiro geral.
```

### Saída do LLM público
```text
Decisão operacional recomendada: Rejeitar o empréstimo.

O perfil apresenta risco elevado em praticamente todos os principais critérios de crédito. O principal ponto crítico é a combinação de CIBIL muito baixo, renda anual insuficiente e valor solicitado alto. A relação empréstimo/renda é elevada, e os ativos declarados não compensam o risco.

Decisão final: Reject.

Uma resposta operacional adequada seria informar que, no momento, a solicitação não atende aos critérios mínimos de crédito, especialmente por causa da capacidade de pagamento e do histórico de crédito.
```

### Observação
A resposta com fine-tuning segue os campos operacionais treinados e o estilo decisório esperado. A resposta do LLM público foi incluída para comparação lado a lado. Esta comparação demonstra especialização de formato e comportamento, não superioridade geral do modelo fine-tuned.

## Caso 3: `pt-case-003`

### Pergunta em português
```text
Qual decisão operacional faz sentido para um cliente autônomo com CIBIL bom, renda média e bom patrimônio bancário?
```

### Entrada normalizada enviada ao modelo
```text
Solicitante tem 1 dependente, é graduado, é autônomo, possui renda anual de 5.200.000, solicitou empréstimo de 11.200.000, prazo de 10 meses, score CIBIL de 704, ativos residenciais de 3.100.000, ativos comerciais de 4.200.000, ativos de luxo de 6.800.000 e ativos bancários de 4.500.000.
```

### Saída histórica esperada
```text
Decisão: aprovar
Motivo: a decisão histórica aprovou este perfil porque o solicitante tem score CIBIL aceitável, exposição moderada do empréstimo em relação à renda e suporte patrimonial suficiente.
Ação recomendada: seguir com a aprovação após validação dos documentos de renda do autônomo e das declarações de ativos.
Mensagem ao cliente: sua solicitação de empréstimo foi aprovada, sujeita à verificação padrão de renda e documentos de suporte.
```

### Saída do modelo base
```text
Vamos analisar esta solicitação de empréstimo com base nas informações fornecidas.

Pontos importantes:

- A aprovação do empréstimo não é garantida: esta análise é apenas informativa. A decisão final depende dos critérios específicos do credor, da tolerância a risco e do perfil financeiro geral do solicitante.
- Faltam informações: precisamos de mais dados para uma avaliação completa. Credores analisam, por exemplo:
  - Relação dívida/renda: quanto da renda do solicitante já está comprometida com dívidas existentes.
  - Relação empréstimo/garantia: comparação entre o valor do empréstimo e o valor da garantia.
  - Histórico profissional: tempo e estabilidade da atividade.
  - Desempenho do negócio: para autônomos, os dados financeiros do negócio são importantes.
  - Histórico de crédito: embora o CIBIL seja um bom ponto de partida, credores também avaliam pagamentos, utilização de crédito e outros fatores.
  - Outros ativos: tipo e valor dos ativos podem influenciar a decisão.

Análise das informações fornecidas:

- Perfil do solicitante: graduado, autônomo, com 1 dependente.
```

### Saída do modelo com fine-tuning
```text
Decisão: aprovar
Motivo: a solicitação foi aprovada porque um score de crédito aceitável, renda compatível com o valor solicitado, forte cobertura por ativos e prazo mais curto, em conjunto, dão suporte ao empréstimo solicitado.
Ação recomendada: seguir com a aprovação, documentando o suporte patrimonial no dossiê.
Mensagem ao cliente: sua solicitação de empréstimo foi aprovada porque a análise encontrou cobertura patrimonial suficiente junto com indicadores financeiros favoráveis.
```

### Saída do LLM público
```text
Decisão operacional recomendada: Aprovar o empréstimo, com validação adicional por ser autônomo.

O perfil é majoritariamente positivo, apesar de haver algum risco por conta da renda autônoma. O CIBIL de 704 indica bom comportamento de crédito, a relação empréstimo/renda é administrável e o patrimônio bancário é relevante. O fato de ser autônomo aumenta o risco operacional, mas não o suficiente para justificar rejeição.

Decisão final: Approve.

Como medida prudencial, faz sentido aprovar condicionando à validação documental da renda, como extratos bancários, declaração fiscal ou comprovantes de faturamento.
```

### Observação
A resposta com fine-tuning segue os campos operacionais treinados e o estilo decisório esperado. A resposta do LLM público foi incluída para comparação lado a lado. Esta comparação demonstra especialização de formato e comportamento, não superioridade geral do modelo fine-tuned.

## Caso 4: `pt-case-004`

### Pergunta em português
```text
Devemos recusar um pedido com CIBIL muito baixo mesmo quando há alguns ativos declarados?
```

### Entrada normalizada enviada ao modelo
```text
Solicitante tem 0 dependentes, é graduado, não é autônomo, possui renda anual de 4.100.000, solicitou empréstimo de 12.300.000, prazo de 16 meses, score CIBIL de 352, ativos residenciais de 2.500.000, ativos comerciais de 1.000.000, ativos de luxo de 3.200.000 e ativos bancários de 900.000.
```

### Saída histórica esperada
```text
Decisão: rejeitar
Motivo: a decisão histórica rejeitou este perfil porque o score CIBIL é muito baixo e o valor solicitado continua significativo em comparação com a renda e o suporte de ativos líquidos.
Ação recomendada: recusar a solicitação e recomendar melhoria do crédito antes de uma nova submissão.
Mensagem ao cliente: não conseguimos aprovar a solicitação de empréstimo neste momento devido à avaliação de risco de crédito.
```

### Saída do modelo base
```text
Vamos analisar esta solicitação de empréstimo com base nas informações fornecidas.

Pontos importantes:

- A aprovação do empréstimo não é garantida: esta análise é apenas informativa. A decisão final depende dos critérios específicos do credor, da tolerância a risco e das condições atuais de mercado.
- Histórico de crédito é fundamental: um CIBIL de 352 sugere risco elevado para o credor.
- Relação empréstimo/renda: este é um indicador crucial. Credores usam essa relação para avaliar a capacidade de pagamento.

Análise:

1. Renda e ativos:
   - Renda anual: 4.100.000 é uma renda relevante, o que é um ponto positivo.
   - Ativos:
     - Residenciais: 2.500.000
     - Comerciais: 1.000.000
     - Luxo: 3.200.000
     - Bancários: 900.000
   - Ativos totais: aproximadamente 7.600.000.
```

### Saída do modelo com fine-tuning
```text
Decisão: rejeitar
Motivo: a solicitação foi rejeitada porque o solicitante tem score de crédito baixo, o que indica maior risco de pagamento.
Ação recomendada: não seguir com a aprovação neste momento; priorizar orientação para fortalecimento do crédito.
Mensagem ao cliente: sua solicitação de empréstimo não foi aprovada com base nos indicadores atuais de crédito e no perfil financeiro geral.
```

### Saída do LLM público
```text
Decisão operacional recomendada: Rejeitar o empréstimo.

Mesmo com alguns ativos declarados, o CIBIL muito baixo pesa fortemente contra a aprovação. O ponto decisivo é o CIBIL de 352, que indica risco de crédito muito elevado. Embora o solicitante tenha renda razoável, não seja autônomo e não tenha dependentes, os ativos declarados não compensam um histórico de crédito tão fraco, especialmente com um valor solicitado de 12.300.000.

Decisão final: Reject.

A resposta operacional mais consistente é recusar o pedido neste momento, ou eventualmente orientar o cliente a solicitar novamente após melhora do histórico de crédito ou com valor menor e garantias mais fortes.
```

### Observação
A resposta com fine-tuning segue os campos operacionais treinados e o estilo decisório esperado. A resposta do LLM público foi incluída para comparação lado a lado. Esta comparação demonstra especialização de formato e comportamento, não superioridade geral do modelo fine-tuned.

## Caso 5: `pt-case-005`

### Pergunta em português
```text
Como tratar um caso com renda muito alta, bom CIBIL e grande volume de ativos para um empréstimo de prazo curto?
```

### Entrada normalizada enviada ao modelo
```text
Solicitante tem 3 dependentes, é graduado, não é autônomo, possui renda anual de 12.800.000, solicitou empréstimo de 25.600.000, prazo de 8 meses, score CIBIL de 812, ativos residenciais de 7.200.000, ativos comerciais de 9.800.000, ativos de luxo de 18.400.000 e ativos bancários de 9.100.000.
```

### Saída histórica esperada
```text
Decisão: aprovar
Motivo: a decisão histórica aprovou este perfil porque o solicitante tem excelente força de crédito, renda anual alta, prazo curto solicitado e cobertura patrimonial substancial.
Ação recomendada: seguir com a aprovação e concluir as verificações finais de documentação.
Mensagem ao cliente: sua solicitação de empréstimo foi aprovada, sujeita à conclusão da documentação e verificação padrão.
```

### Saída do modelo base
```text
Vamos analisar esta solicitação de empréstimo com base nas informações fornecidas.

Pontos importantes:

- Tipo de empréstimo: precisaríamos saber o tipo de empréstimo para entender os critérios específicos. É um empréstimo pessoal, empresarial, imobiliário etc.?
- Histórico de crédito: um CIBIL de 812 geralmente é considerado bom, mas é importante entender o histórico completo do solicitante.
- Relação empréstimo/renda: este é um fator-chave na aprovação. Ele compara o valor do empréstimo com a renda anual.
- Relação dívida/renda: mede o percentual da renda comprometido com dívidas.
- Finalidade do empréstimo: o motivo do empréstimo influencia a decisão do credor.

Análise:

Pontos fortes:

- Renda alta: o solicitante tem renda anual significativa de 12.800.000. Isso é um forte indicador de capacidade de pagamento.
- Ativos robustos: o solicitante tem ativos substanciais, incluindo residenciais, comerciais, de luxo e bancários.
```

### Saída do modelo com fine-tuning
```text
Decisão: aprovar
Motivo: a solicitação foi aprovada porque o solicitante tem score de crédito forte, boa capacidade de pagamento em relação ao valor solicitado e forte cobertura por ativos.
Ação recomendada: seguir com a aprovação e revisão padrão da documentação.
Mensagem ao cliente: sua solicitação de empréstimo foi aprovada com base no seu perfil de crédito e na análise financeira geral da solicitação.
```

### Saída do LLM público
```text
Decisão operacional recomendada: Aprovar o empréstimo.

O caso apresenta um perfil de crédito muito favorável, com renda alta, CIBIL forte, bom volume de ativos e prazo curto. A combinação de CIBIL 812, renda anual elevada e ativos totais muito superiores ao valor solicitado indica baixo risco operacional. O prazo curto também favorece a aprovação, desde que a capacidade mensal de pagamento seja compatível com a política de crédito.

Decisão final: Approve.

A aprovação faz sentido, com validações documentais padrão de renda, vínculo e ativos declarados.
```

### Observação
A resposta com fine-tuning segue os campos operacionais treinados e o estilo decisório esperado. A resposta do LLM público foi incluída para comparação lado a lado. Esta comparação demonstra especialização de formato e comportamento, não superioridade geral do modelo fine-tuned.
