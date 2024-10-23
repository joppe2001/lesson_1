# base_plotter.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
import os
from dataclasses import dataclass


@dataclass
class DimReductionData:
    """Container for dimensionality reduction results."""
    embedded_data: np.ndarray
    labels: np.ndarray
    explained_variance: Optional[float] = None



class BasePlotter:
    """Utility class for creating consistent visualizations across the project"""

    STYLE_PRESETS = {
        'default': {
            'style': 'seaborn-v0_8-darkgrid',
            'figure_size': (12, 8),
            'title_size': 14,
            'dpi': 300,
            'colors': {
                'positive': 'lightgreen',
                'negative': 'lightcoral',
                'neutral': 'lightgray',
                'line': 'black'
            }
        },
        'minimal': {
            'style': 'seaborn-v0_8-whitegrid',
            'figure_size': (10, 6),
            'title_size': 12,
            'dpi': 300,
            'colors': {
                'positive': '#90EE90',
                'negative': '#F08080',
                'neutral': '#D3D3D3',
                'line': '#333333'
            }
        },
        'dark': {
            'style': 'seaborn-v0_8-dark',
            'figure_size': (12, 8),
            'title_size': 14,
            'dpi': 300,
            'colors': {
                'positive': '#00FF00',
                'negative': '#FF0000',
                'neutral': '#808080',
                'line': '#FFFFFF'
            }
        }
    }

    def __init__(self,
                 preset: str = 'default',
                 figure_size: Optional[Tuple[int, int]] = None,
                 dpi: Optional[int] = None,
                 style: Optional[str] = None):
        """
        Initialize the plotter with given settings or preset.

        Args:
            preset: Style preset ('default', 'minimal', or 'dark')
            figure_size: Optional override for figure size
            dpi: Optional override for DPI
            style: Optional override for matplotlib style
        """
        self.settings = self.STYLE_PRESETS[preset].copy()

        if figure_size:
            self.settings['figure_size'] = figure_size
        if dpi:
            self.settings['dpi'] = dpi
        if style:
            self.settings['style'] = style

        plt.style.use(self.settings['style'])

    def setup_figure(self, title: str) -> Tuple[plt.Figure, plt.Axes]:
        """Create and setup a new figure with consistent styling"""
        fig, ax = plt.subplots(figsize=self.settings['figure_size'])
        fig.suptitle(title, fontsize=self.settings['title_size'])
        return fig, ax

    def save_plot(self, path: str, tight: bool = True) -> None:
        """Save the current plot to file with consistent settings"""
        if tight:
            plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=self.settings['dpi'], bbox_inches='tight')
        plt.close()

    def create_time_series(self,
                           data: Union[pd.Series, pd.DataFrame],
                           title: str,
                           ylabel: str,
                           columns: Optional[List[str]] = None,
                           output_path: Optional[str] = None) -> None:
        """
        Create a time series plot with consistent styling.

        Args:
            data: Series or DataFrame to plot
            title: Plot title
            ylabel: Y-axis label
            columns: List of columns to plot (if DataFrame)
            output_path: Optional path to save the plot
        """
        fig, ax = self.setup_figure(title)

        # Handle different input types
        if isinstance(data, pd.Series):
            plot_data = data
            if isinstance(data.index, pd.PeriodIndex):
                plot_index = data.index.astype('datetime64[ns]')
            else:
                plot_index = data.index

            # Plot line
            line = ax.plot(plot_index, plot_data.values,
                           color=self.settings['colors']['line'],
                           linewidth=2)

            # Add zero line if data contains positive and negative values
            if (plot_data > 0).any() and (plot_data < 0).any():
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

                # Fill between
                ax.fill_between(plot_index, plot_data.values, 0,
                                where=(plot_data.values > 0),
                                color=self.settings['colors']['positive'],
                                alpha=0.3)
                ax.fill_between(plot_index, plot_data.values, 0,
                                where=(plot_data.values <= 0),
                                color=self.settings['colors']['negative'],
                                alpha=0.3)

        else:  # DataFrame
            if columns is None:
                columns = data.columns
            for column in columns:
                ax.plot(data.index, data[column], label=column)
            ax.legend()

        # Customize
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45)

        if output_path:
            self.save_plot(output_path)

    def create_stacked_bar(self,
                           data: pd.DataFrame,
                           labels: List[str],
                           title: str,
                           xlabel: str,
                           horizontal: bool = True,
                           output_path: Optional[str] = None) -> None:
        """Create a stacked bar chart with consistent styling"""
        fig, ax = self.setup_figure(title)

        positions = np.arange(len(data))

        if horizontal:
            plot_func = ax.barh
            ax.set_yticks(positions)
            if isinstance(data.index, (pd.Index, pd.MultiIndex)):
                ax.set_yticklabels(data.index)
        else:
            plot_func = ax.bar
            ax.set_xticks(positions)
            if isinstance(data.index, (pd.Index, pd.MultiIndex)):
                ax.set_xticklabels(data.index, rotation=45)

        left = np.zeros(len(data))
        for label in labels:
            plot_func(positions, data[label], left=left, label=label)
            left += data[label]

        ax.set_xlabel(xlabel)
        ax.legend()

        if output_path:
            self.save_plot(output_path)

    def create_distribution(self,
                            data: pd.Series,
                            title: str,
                            xlabel: str,
                            ylabel: str = 'Frequency',
                            bins: int = 50,
                            kde: bool = True,
                            output_path: Optional[str] = None) -> None:
        """Create a distribution plot with consistent styling"""
        fig, ax = self.setup_figure(title)
        sns.histplot(data=data, bins=bins, kde=kde, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if output_path:
            self.save_plot(output_path)

    def create_heatmap(self,
                       data: pd.DataFrame,
                       title: str,
                       cmap: str = 'YlOrRd',
                       annot: bool = True,
                       fmt: str = '.2f',
                       output_path: Optional[str] = None) -> None:
        """Create a heatmap with consistent styling"""
        fig, ax = self.setup_figure(title)
        sns.heatmap(data, cmap=cmap, annot=annot, fmt=fmt, ax=ax)

        if output_path:
            self.save_plot(output_path)

    def create_barchart(self,
                        data: pd.DataFrame,
                        title: str,
                        xlabel: str,
                        ylabel: str,
                        output_path: Optional[str] = None) -> None:
        """Create bar chart with consistent styling"""
        fig, ax = self.setup_figure(title)

        # Ensure we're using the correct column names from the data
        sns.barplot(
            data=data,
            x='Author',  # Use actual column name from DataFrame
            y='Percentage',  # Use actual column name from DataFrame
            ax=ax
        )

        # Set custom labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if output_path:
            self.save_plot(output_path)

            # Add this method at the end of the BasePlotter class

    def create_dim_reduction_plot(self,
                                  data: DimReductionData,
                                  title: str,
                                  alpha: float = 0.6,
                                  point_size: int = 100,
                                  add_legend: bool = True,
                                  output_path: Optional[str] = None) -> None:
        fig, ax = self.setup_figure(title)

        # Create scatter plot with unique colors for each label
        scatter = ax.scatter(
            data.embedded_data[:, 0],
            data.embedded_data[:, 1],
            c=pd.factorize(data.labels)[0],
            alpha=alpha,
            s=point_size,
            cmap='tab20'  # Use a colormap that works well with multiple categories
        )

        # Add legend if requested
        if add_legend:
            legend = ax.legend(
                scatter.legend_elements()[0],
                np.unique(data.labels),
                title="Labels",
                loc="upper right"
            )
            ax.add_artist(legend)

        # Add explained variance if available
        if data.explained_variance is not None:
            subtitle = f'Explained Variance: {data.explained_variance:.2%}'
            ax.text(0.02, 0.98, subtitle,
                    transform=ax.transAxes,
                    verticalalignment='top')

        # Set labels
        ax.set_xlabel('First Component')
        ax.set_ylabel('Second Component')

        # Save if path provided
        if output_path:
            self.save_plot(output_path)

    def create_boxplot(self,
                       data: pd.DataFrame,
                       x: str,
                       y: str,
                       title: str,
                       ylabel: str,
                       palette: str = 'husl',
                       output_path: Optional[str] = None) -> None:
        """
        Create a boxplot with consistent styling.

        Args:
            data: DataFrame containing the data
            x: Column name for x-axis (categories)
            y: Column name for y-axis (values)
            title: Plot title
            ylabel: Y-axis label
            palette: Color palette for the boxes
            output_path: Optional path to save the plot
        """
        fig, ax = self.setup_figure(title)

        # Create boxplot using seaborn
        sns.boxplot(
            data=data,
            x=x,
            y=y,
            ax=ax,
            palette=palette,
            width=0.7,
            fliersize=5
        )

        # Customize
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45)

        # Add median values on top of each box
        medians = data.groupby(x)[y].median()
        for i, median in enumerate(medians):
            ax.text(
                i,
                median,
                f'{median:.2f}',
                horizontalalignment='center',
                verticalalignment='bottom',
                fontweight='bold'
            )

        if output_path:
            self.save_plot(output_path)