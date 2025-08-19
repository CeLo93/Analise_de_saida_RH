<p align="center"> 
👩‍💼 <b> ANÁLISE DE SAÍDA DE COLABORADORES DE EMPRESA </b> 👩‍💼
<p>




Fiz este modelo de Machine Learning para analisar a possibilidade de um funcionário sair ou não de uma empresa, com base nos atributos levantados pelo RH da empresa (conjunto de dados fictício criado por cientistas de dados da IBM). Todos os códigos e demais gráficos estão no notebook acima. Também procurei comentar ao máximo, no andamento do modelo, o que fui fazendo e o porquê. Assim, venho aqui mostrar um resumo dos resultados obtidos através de toda análise e tratamento dos dados até a sua aplicação nos algoritmos. Mostrando alguns insights feitos através do modelo.
 
  Colab 💻 : https://drive.google.com/file/d/1ROYLNDTJhHyXlRSRI620E1h0HMvHgmn7/view?usp=sharing

Dataset 📑 : https://drive.google.com/file/d/13bnw8KaNiATySh41J_WHfnkoZzSBiJia/view?usp=sharing
  
Attrition: YES = 1; NO = 0
  
Attrition quer dizer se o funcionário saiu (saíra, no caso da previsão) da empresa ou não.

  <p align="center"> 
📊 <b> ANÁLISE DOS RESULTADOS  </b> 📊
<p>
 
 ![g2](https://github.com/CeLo93/Analise_de_saida_RH/assets/92175791/6c188941-2ff2-47ae-b2d5-9c85f042fa67)
 <p align="center"> Gráfico 01: Matriz de Correlação <p>

É deveras interessante visualizar a matriz de correlação e ver como os atributos se correlacionam entre si, por exemplo: o Job Level (cargo da pessoa na empresa) com Total Working Years (tempo de trabalho) ou o próprio Total Working Years com Monthly Income (renda mensal). Esta análise é muito importante para qualquer modelo, pois é possível extrair diversos insights para análise na empresa.
 
  ![g1](https://github.com/CeLo93/Analise_de_saida_RH/assets/92175791/ea46a1dc-9030-4ab3-9dcb-10a82f0a6c1c)
 <p align="center"> Gráfico 02: Gráfico de saída por idade <p>
   
  
Já o gráfico do Attrition pela idade mostra algo, de certa forma, óbvio, mas muito interessante. Observa-se que os colaboradores de 18 à 20 anos tendem a sair da empresa, talvez por quererem oportunidades melhores. Mas, observa-se que pessoas na faixa dos 30 à 45 tendem a permanecer na empresa. Muito por conta de busca de estabilidade ou o próprio tempo na empresa e, consequentemente, um cargo melhor remunerado (esta correlação vista na matriz de correlação, na imagem anterior).
  
 ![g3](https://github.com/CeLo93/Analise_de_saida_RH/assets/92175791/9631f14c-e3a2-484e-80e4-76fab4399931)
 <p align="center"> Gráfico 03: Gráfico geral <p>

Analisando a Imagem 03, temos uma análise de quantas pessoas saem por departamento. Essa é uma análise muito útil, pois com essa análise a empresa pode direcionar um estudo para aquele setor com maior incidência de saídas (Sales Represantative, por exemplo). Dessa forma, podemos fazer um DataFrame específico para o departamento, fazendo uma análise mais detalhada. Também vemos, pelo gráfico do estado civil, que pessoas solteiras tendem a sair mais da empresa do que as casadas, o que também parece ser lógico, visto as pessoas casadas terem uma, digamos, necessidade de estabilidade maior. Temos também o gráfico de envolvimento no trabalho, que foi uma pesquisa feita pelo RH quanto ao grau de envolvimento da pessoa no trabalho, com um grau de 1-4. O que apresenta uma maior incidência de pessoas que permaneceram no trabalho com o grau 3, muito por conta de ter um equilíbrio entre o extremo do “hard working” e o “quiet quitting”. Já na última análise dos gráficos dessa figura, temos a incidência de saídas pelo cargo na empresa. E notamos que, obviamente, temos menos pessoas com cargos mais elevados, mas também temos poucas saídas. Isso se dá muito pelo tempo de trabalho da empresa, experiência de trabalho e tudo isso culminando a uma “escalada” em cargos superiores e, consequentemente, melhores remunerações. Já os cargos mais baixos, temos muito mais colaboradores, porém mais saídas da empresa também. Muito em conta de novas oportunidades em outras empresas ou, como veremos no próximo gráfico, insatisfação com a remuneração e projeção no cargo atual.

 ![g4](https://github.com/CeLo93/Analise_de_saida_RH/assets/92175791/94336bd5-427e-4641-8b3b-f44979596b46)
 <p align="center"> Gráfico 04: Gráfico de renda mensal pelo cargo <p>

Neste gráfico de renda mensal pelo cargo temos a análise entre a renda mensal e o departamento em que a pessoa atua. Como vimos na análise anterior, o cargo de "Sales Representative" tem uma incidência maior de saída dos colaboradores, em relação aos outros departamentos. O que podemos observar é que este cargo possui uma faixa salarial com muita pouca dilatação (sem considerar os outliers, que podem ser comissões ou bonificações para outros colaboradores). Assim, os colaboradores desta função podem ter um sentimento de pouca projeção de carreira e, consequentemente, uma desmotivação, tendo em vista a pouca possibilidade de crescimento dentro da área, na empresa, como se vê no eixo representando a faixa de remuneração mensal. Uma boa explicação, para os outliers do departamento "Sales Representative", pode ser dada pelo fato de termos funcionários com um ótimo desempenho e, como vemos na matriz de correlação anterior, uma boa taxa de aumento salarial, mas, por algum motivo, esses colaboradores não foram promovidos à "Sales Executive".  O que se vê, também, é o padrão visto nos outros gráficos, ou seja, vemos, através da matriz de correlação, que pessoas com mais tempo de experiência de trabalho terá, na empresa, rendas melhores e cargos melhores. O que, indiretamente está relacionado entre a classificação de desempenho e a porcentagem de aumento salarial, em uma primeira análise e como foi dito no caso de promoção à "Sales Executive".

Quanto aos algoritmos: 
  
No modelo utilizo três tipos de análises algorítmicas: Regressão logística, Random forest e Redes neurais artificiais. No modelo, após todo o processo de análise e tratamento, o algoritmo que mais se adaptou, quanto a sua precision e recall, apesar de ter uma accuracy acima de 88% em ambos, foi o de Regressão logística. Um detalhe importante é que pode haver uma pequena discrepância, entre os resultados, pois o algoritmo usa parâmetros que são inicializados aleatoriamente, então uma pequena variação é esperada. Muito provavelmente, ao rodar o modelo mais vezes, você notará essas pequenas discrepâncias nos resultados. Assim, nas vezes que testei, a Regressão logística funcionou melhor. Resumindo, a regressão logística é um algoritmo utilizado para classificação, apesar de ter a palavra regressão em seu nome. Essa nomenclatura está relacionada com o fato da Regressão Logística ser construída a partir da aplicação de uma transformação/função (denominada função sigmoide) sobre a Regressão Linear. Essa função sigmoide recebe um número real (combinação linear de variáveis) como entrada e retorna uma saída dentro do intervalo [0, 1], que representa a probabilidade do registro de entrada pertencer a classe evento, no caso sair ou não da empresa.
 
   ![sasasa](https://github.com/CeLo93/Analise_de_saida_RH/assets/92175791/9a5e2589-caec-441e-ba06-fc594e92743b)
 <p align="center"> Gráfico 05 – Comparação entre as regressões <p>


![res](https://github.com/CeLo93/Analise_de_saida_RH/assets/92175791/f40f3d11-e5aa-4fda-8b24-1198a7204a85)

Após salvar o classificador, simulamos um novo funcionário e testamos a aplicação do modelo a este novo funcionário. O resultado mostra que este funcionário, baseado no modelo desenvolvido e o meu teste, terá 68,11% de probabilidade de sair da empresa, com uma chance de 44% de esta previsão estar correta.

Como este foi um estudo de caso, baseado em um conjunto de dados fictício criado por cientistas de dados da IBM, como dito anteriormente, apesar do modelo ter uma accuracy excelente, analisar os indicadores de precison e recall mostram que o modelo não tem uma classificação tão equilibrada para classificar quem vai sair da empresa (1) em detrimento de quem irá ficar (0). 
Por ter sido um estudo de caso, parei por aqui nesta análise. Mas, muito provavelmente, eliminando alguns atributos com pouca correlação do modelo , fazendo um estudo específico para cada departamento da empresa filtrando, assim, melhores resultados e trazendo para este DataFrame, poderíamos ter uma assertividade maior.
