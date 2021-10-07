# %%
import pandas as pd
import logomaker

# %%
def plot_sequence_attribution(attribution_matrix, sigma=['A', 'C', 'G', 'T'], width=20, height=4):
    attribution_df = pd.DataFrame(attribution_matrix, columns=sigma)

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
    attribution_logo.fig.set_figheight(height)
    attribution_logo.fig.set_figwidth(width)

    return attribution_logo