# Três casos de referência para os relatórios independentes

Estes casos são inteiramente fictícios. Permitem trabalhar em prosecution,
defense e na passagem do contexto pelo orchestrator enquanto o pré-processamento
está a ser alterado. Não representam clientes reais nem classes de verdade
«culpado» e «inocente». Ainda não introduzem discussão entre os agentes.

## Começar aqui

1. Ler `01_movimentos_concentrados.json`, primeiro o bloco `input`.
2. Ler `evaluation` e procurar nos dados o suporte para cada critério.
3. Correr os testes offline abaixo antes de alterar prompts ou relatórios.

| Ficheiro | Situação | O que observar nos relatórios |
| --- | --- | --- |
| `01_movimentos_concentrados.json` | Três entradas de 5000 USD e uma saída de 14900 USD em 15 minutos | Prosecution relaciona os factos. Defense reconhece o padrão e as explicações ainda por confirmar. |
| `02_pagamentos_regulares.json` | Três pagamentos mensais de 1200 USD entre as mesmas contas | Defense sustenta a regularidade com evidência. Prosecution reconhece a limitação dos indícios. |
| `03_informacao_insuficiente.json` | Apenas data, score e sinal de alerta | Ambos reconhecem os limites e identificam os dados em falta. |

## Porque estão organizados assim

Cada JSON contém `input`, com os argumentos de `build_case_context`, e
`evaluation`, com os comportamentos esperados e conclusões proibidas.
`rationale` explica a escolha do cenário. O módulo `__init__.py` envia apenas
`input` para o contexto. As respostas esperadas ficam fora do prompt para evitar
que o modelo copie os critérios da avaliação.

Todos têm `alert: true` porque os revisores atuais só aceitam alertas sinalizados.
Os scores foram escolhidos manualmente e não são probabilidades verificadas de
crime. Não estamos a testar o limiar, o top-k diário nem a deteção do modelo.
As linhas incluem a transação alertada e, quando existem, movimentos relacionados.
O campo `row_id` identifica cada linha fictícia. Os montantes estão em USD para
evitar introduzir conversões cambiais neste primeiro exercício.

O segundo caso permite uma interpretação legítima, mas não a comprova. Não há
contrato ou finalidade de pagamento nos dados. No terceiro, `transactions: []`
significa histórico não fornecido. O contrato atual não transmite explicitamente
a cobertura do histórico. A ausência de evidência no contexto não permite
afirmar que não existem outras transações.

## Testes offline, sem API

Executar na raiz do projeto, com as dependências instaladas no ambiente virtual.
Se o ambiente tiver outro nome, substituir o caminho do executável.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s agents\tests -p "test_reference_cases.py" -v
.\.venv\Scripts\python.exe -B -m unittest discover -s agents\tests -v
```

Os testes verificam os dados essenciais, IDs estáveis, omissão de campos em falta
e receção da mesma evidência pelos dois revisores. Usam respostas simuladas e
uma barreira contra chamadas reais. Passarem não demonstra qualidade dos
relatórios gerados por um LLM. Voltar a correr ambos os comandos depois de
alterar casos, prompts, contratos ou o orchestrator.

Para inspecionar a evidência sem chamar um modelo, usar Python na raiz do projeto.

```python
from agents.reference_cases import build_reference_context

context = build_reference_context("01_movimentos_concentrados")
for item in context.evidence:
    print(item.id, item.description)
```

## Avaliação posterior com um modelo real

O exemplo seguinte faz duas chamadas ao fornecedor configurado em `.env` e pode
ter custos. Só o executar quando se pretender avaliar relatórios reais. Exige
`PROVIDER`, a chave correspondente e um `MODEL` disponível na conta. Esta entrega
valida o fluxo offline e não verifica a disponibilidade dos modelos configurados.

```python
from dataclasses import asdict
import json
from agents.orchestrator import review_case
from agents.reference_cases import build_reference_context

context = build_reference_context("01_movimentos_concentrados")
result = review_case(context)
print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
```

Repetir para os três nomes de ficheiro, sem `.json`, e avaliar os seis relatórios
independentes. Não exigir frases idênticas às dos critérios. Para cada relatório,
registar caso, papel, data, fornecedor, modelo, versão do código e texto produzido.
Para cada ponto abaixo, registar «cumpre», «parcial» ou «falha», com um exemplo
do relatório e os IDs relevantes.

- Os critérios do respetivo papel em `evaluation` estão cobertos?
- Cada afirmação factual tem citações que a suportam realmente?
- Os cálculos e relações entre datas, contas e montantes estão corretos?
- O relatório distingue factos, hipóteses e informação em falta?
- Evita todas as conclusões em `forbidden` e respeita `limitations`?

Como critério inicial, um relatório passa quando todos estes pontos cumprem.
Uma afirmação inventada, uma citação sem suporte ou uma conclusão proibida
impede a aprovação. Guardar os resultados antes e depois de mudar os prompts
permite perceber se melhoraram. A validação automática existente verifica IDs
e formato, mas a verificação semântica continua a exigir leitura humana.

O caso insuficiente pode usar um finding que explique os limites do alerta e cite
o score/alerta recebido. Essas referências delimitam os dados disponíveis, mas
não provam a ausência de histórico externo. Não é preciso inventar um campo de
resposta nem um veredicto para satisfazer o contrato atual.

Estes três casos são o primeiro conjunto de desenvolvimento. Não são uma avaliação
representativa nem substituem um conjunto maior e independente antes de concluir
que os agentes estão prontos. A discussão entre agentes fica para uma fase posterior.
