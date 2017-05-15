# Rede-Neural-Simples-Java
Rede Neural


Funcionalidades:
•	Possibilidade de informar número de neurônios da camada de entrada;
•	Possibilidade de informar número de neurônios da camada de intermediária;
•	Possibilidade de informar número de neurônios da camada de saída;
•	Possibilidade de carregar dados de treinamento a partir de um arquivo CSV;
•	Possibilidade de informar o número de treinamentos;
•	Possibilidade de efetuar treinamento da rede neural;

Resultados:
•	Retornar a taxa de erro atual;
•	Número de treinamentos realizados;
•	Tempo de treinamento;

Testes da Rede Neural:
•	É informado os dados para testar a rede;
•	Retorna a informação de cada neurônico de saída e o valor de ativação;

Descrição do Problema:
	Foi utilizado o algoritmo backpropagation para efetuar o treinamento da rede neural, primeiramente definimos um padrão a camada de entrada da rede, as atividades vão fluindo através da rede, camada por cada, até que o resultado seja produzido pela camada de saída. Num segundo momento o resultado de saída e comparado com a saída desejada, caso o resultado não esteja correto o erro é calculado e propagado a partir da saída até a camada de entrada modificando assim os pesos de conexões.

 
Arquitetura da solução:
	A solução da rede foi criada em modo flexível onde é possível definir neurônios da camada de entrada, intermediária e saída mediante a configuração da aplicação. Como também os dados de treinamento são carregadores a partir de um CSV.
	O problema arquitetado foi da função matemática onde a saída da rede representa um número.
Note os dados de treinamento abaixo a entrada 0,0,0,1 deve ativar somente o primeiro neurônio, este representando o número 1.
 
	A rede foi treinada 800 vezes num tempo de 295 ms (somente treinamento da rede) e apresentando uma taxa de erro de 0,000740.
	
