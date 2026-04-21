import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# É necessário instalar o pacote jinja2 (pip install jinja2)
def df_summary_report(df):
    """
    Descreve valores estatísticos da dataframe.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    Retorno
    -------
    pd.Series
        Valores estatísticos e quantitativos estilizados.
    """

    moda = df.mode().iloc[0]

    mem_consumo = df.memory_usage(deep=True, index=False)/1024**2
    total = sum(mem_consumo)

    freq = df.apply(
        lambda col: col.value_counts(dropna=True).iloc[0]
        if not col.dropna().empty else None
    )

    resumo = pd.DataFrame({
        'Coluna': df.columns,            #Lista os nomes das colunas do DataFrame.
        'Tipo': df.dtypes.values,        #Retorna o tipo de dado (dtype) de cada coluna.
        'Quantidade de Dados Não Vazios': df.notna().sum().values,  #Conta a quantidade de valores não nulos por coluna.
        'Quantidade de Dados Vazios': df.isna().sum().values,       #Conta a quantidade de valores nulos (NaN) por coluna.
        'Valores Únicos': df.nunique().values,       #Conta a quantidade de valores únicos de cada coluna
        'Valor mais Frequente': moda,    #Calcula a moda de cada coluna    
        'Frequência': freq,               #Mostra a frequência da moda
        'Porcentagem de Unicidade': ((df.nunique() / len(df)) * 100).round(2).values,    #Cardinalidade - baixa cardinalidade indica alta repetição; alta cardinalidade indica baixa repetição
        'Porcentagem de Valor Vazios (%)': (df.isna().mean() * 100).round(2).values,  #Calcula a porcentagem de valores nulos por coluna.
        f'Consumo de Memória - Total: {total:.2f} (MB)': mem_consumo    #Calcula a quantidade de memória usada pelo dataset
    }).reset_index(drop=True)

    #colormaps
    cmap_vazios = sns.light_palette("#BD2A2E", as_cmap=True)
    cmap_unicidade = sns.light_palette("#13678A", as_cmap=True)
    cmap_unique = sns.light_palette("#1f77b4", as_cmap=True)
    cmap_freq = sns.light_palette("#13678A", as_cmap=True)

    styled = (resumo.style
        .set_properties(**{
            'background-color': "#101719",
            'color': '#E0E0E0',  
            'border': '1px solid #2F3D40',
            'text-align': 'center'
        })
        .background_gradient(subset=['Porcentagem de Valor Vazios (%)'], cmap=cmap_vazios, vmin=0, vmax=100)
        .background_gradient(subset=['Porcentagem de Unicidade'], cmap=cmap_unicidade, vmin=0, vmax=100)
        .background_gradient(subset=["Valores Únicos"],cmap=cmap_unique)
        .background_gradient(subset=["Frequência"],cmap=cmap_freq)
        .background_gradient(subset=[f"Consumo de Memória - Total: {total:.2f} (MB)"], cmap=cmap_freq)
        .bar(subset=['Quantidade de Dados Vazios'], color="#BD2A2E")
        .format({'Porcentagem de Valor Vazios (%)': '{:.2f}', 'Porcentagem de Unicidade': '{:.2f}'})
        .set_table_styles([{
                'selector': 'th',
                'props': [
                    ('background-color', "#0c2845"),
                    ('color', 'white'),
                    ('text-align', 'center'),
                    ('font-size', '13px')
                ]
            }
        ])
        .set_properties(
            subset=pd.IndexSlice[:, resumo.columns[0]],**{
                'background-color': '#012030',
                'font-weight': 'bold',
                'color': '#FFFFFF'
            })
        
    )
    return styled
#----------------------------------------------------------------------------------------------------------------------
def plot_origem_destino_map(
    df,
    lat_origem,
    lon_origem,
    lat_destino,
    lon_destino
):
    """
    Plota pontos de origem e destino em um mapa interativo usando Plotly.

    Parâmetros:
    - df (pd.DataFrame): DataFrame com os dados
    - lat_origem (str): coluna latitude da origem
    - lon_origem (str): coluna longitude da origem
    - lat_destino (str): coluna latitude do destino
    - lon_destino (str): coluna longitude do destino
    - zoom (float): nível de zoom do mapa
    - show (bool): se True, exibe o mapa

    Retorno:
    - fig (plotly.graph_objs.Figure)
    """

    df_origem = df[[lat_origem, lon_origem]].copy()
    df_origem["Legenda"] = "Origem - Restaurante"
    
    df_origem = df_origem.rename(columns={
        lat_origem: "lat",
        lon_origem: "lon"
    })

    df_destino = df[[lat_destino, lon_destino]].copy()
    df_destino["Legenda"] = "Destino - Local de Entrega"
    
    df_destino = df_destino.rename(columns={
        lat_destino: "lat",
        lon_destino: "lon"
    })

    df_plot = pd.concat([df_origem, df_destino], ignore_index=True)

    fig = px.scatter_map(
        df_plot,
        lat="lat",
        lon="lon",
        color="Legenda",
        zoom=1.35
    )

    fig.update_layout(
        map_style = "carto-positron",
        margin = {"r": 0, "t": 0, "b": 0, "l": 0}
    )

    fig.show()
#--------------------------------------------------------------------------------------
def plot_delivery_age_analysis(df, col_rating, col_age):
    """
    Plota:
    1) Mediana da idade dos entregadores por rating
    2) Distribuição da idade dos entregadores

    Parâmetros:
    - df (pd.DataFrame): DataFrame com os dados
    - col_rating (str): coluna de ratings
    - col_age (str): coluna de idade

    Retorno:
    - fig, ax
    """

    # Agregação
    df_deli_age = df.groupby(col_rating)[col_age].median()

    # Criação dos subplots
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # Gráfico 1 — Mediana da idade por rating
    df_deli_age.plot(
        kind='line',
        ax=ax[0],
        color = '#103778',
        linewidth=2.5,
        marker='o'
    )

    ax[0].set_title('Mediana da Idade por Rating', fontsize=14, pad=10)
    ax[0].set_xlabel('Ratings')
    ax[0].set_ylabel('Idade')
    ax[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Gráfico 2 — Distribuição de idades
    sns.histplot(
        df[col_age],
        kde=True,
        bins=20,
        color = '#103778',
        edgecolor='black',
        alpha=0.8,
        ax=ax[1]
    )

    ax[1].set_title('Distribuição por Idade dos Deliveries')
    ax[1].set_xlabel('Idade')
    ax[1].set_ylabel('Frequência')

    plt.tight_layout()
    plt.show()
#-------------------------------------------------------------------------
def plot_proportion_bar(df, column, title, xlabel):
    """
    Plota um gráfico de barras com a proporção percentual de uma variável categórica.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    column : str
        Coluna categórica a ser analisada.
    title : str
        Título do Grafico
    xlabel : str
        Label do eixo x

    Retorno
    -------
    pd.Series
        Série com as proporções (%) por categoria.
    """
    
    palette = ['#151F30', '#103778', '#0593A2', '#FF7A48', '#E3371E']

    # Cálculo da proporção
    proportions = df[column].value_counts(normalize=True) * 100

    # Plot
    plt.figure(figsize=(10, 5))
    plt.style.use('ggplot')
    plt.bar(
        proportions.index.astype(str),
        proportions.values,
        color=palette[:len(proportions)]
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Proporção (%)', fontsize=14)

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Valores acima das barras
    for i, valor in enumerate(proportions.values):
        plt.text(
            i,
            valor + 0.3,
            f'{valor:.2f}%',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    plt.show()
#------------------------------------------------------------------------------
def plot_rating_categoria_analise(df, col_categoria, col_rating, col_tempo):
    """
    Cria três gráficos:
    1) Distribuição percentual por categoria (barra)
    2) Distribuição de ratings (histograma)
    3) Densidade do tempo de entrega por categoria (KDE)

    Layout:
    - Dois gráficos na primeira linha
    - Um gráfico centralizado na segunda linha
    """

    # Preparação dos dados
    cont_percen = df[col_categoria].value_counts(normalize=True) * 100
    rating_groups = df.groupby(col_categoria)[col_tempo]

    # Layout com GridSpec
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])

    ax1 = fig.add_subplot(gs[0, 0])   # Topo esquerda
    ax2 = fig.add_subplot(gs[0, 1])   # Topo direita
    ax3 = fig.add_subplot(gs[1, :])   # Segunda linha (centralizado)

    # Gráfico 1 — Barras (%)
    cont_percen.plot(
        kind='bar',
        color=['#151F30', '#103778', '#0593A2', '#FF7A48', '#E3371E'],
        ax=ax1
    )

    ax1.set_title('Distribuição por Categoria')
    ax1.set_xlabel('Categoria')
    ax1.set_ylabel('Percentual %')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

    for i, valor in enumerate(cont_percen):
        ax1.text(
            i,
            valor + 0.2,
            f'{valor:.2f}%',
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    # Gráfico 2 — Histograma
    sns.histplot(
        df[col_rating],
        kde=True,
        bins=20,
        color='#0593A2',
        ax=ax2
    )

    ax2.set_title('Distribuição por Ratings / Delivery')
    ax2.set_xlabel('Ratings')
    ax2.set_ylabel('Frequência')

    # Gráfico 3 — KDE por categoria
    cores = ['#151F30', '#103778', '#0593A2', '#FF7A48']

    for (group, values), cor in zip(rating_groups, cores):
        sns.kdeplot(
            values,
            label=f'Grupo {group}',
            fill=True,
            color=cor,
            ax=ax3
        )

    ax3.set_title('Distribuição do Tempo Gasto por Taxa de Avaliação', fontsize=14)
    ax3.set_xlabel('Time Taken (min)')
    ax3.set_ylabel('Densidade')
    ax3.legend(title='Avaliação dos Grupos')

    plt.tight_layout()
    plt.show()
