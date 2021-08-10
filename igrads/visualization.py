# %%
import pandas as pd
import logomaker

# %%
def plot_sequence_attribution(attribution_matrix):
    attribution_df = pd.DataFrame(attribution_matrix, columns=['A', 'C', 'G', 'T'])

    # create Logo object
    attribution_logo = logomaker.Logo(attribution_df,
                            shade_below=.5,
                            fade_below=.5,
                            font_name='Arial Rounded MT Bold')

    # style using Logo methods
    attribution_logo.style_spines(visible=False)
    attribution_logo.style_spines(spines=['left', 'bottom'], visible=True)
    #attribution_logo.style_xticks(rotation=90, fmt='%d', anchor=0)

    # style using Axes methods
    attribution_logo.ax.set_ylabel("IG Attribution", labelpad=-1)
    #attribution_logo.ax.xaxis.set_tick_params(which='both', bottom=False, top=False, labelbottom=False)

    # adjust figure width and heigh
    logo.fig.set_figheight(3)
    logo.fig.set_figwidth(20)

    return attribution_logo