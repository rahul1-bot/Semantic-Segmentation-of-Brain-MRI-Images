from __future__ import annotations
from dataset import * 


class BrainAnalysis:
    def __init__(self, brain_metaData: pd.DataFrame, brain_df: pd.DataFrame) -> None:
        self.brain_metaData = brain_metaData
        self.brain_df = brain_df
        
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'Object_ID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
        
    
    def __str__(self) -> str(dict[str, list[str]]):
        dataKey_map: dict[str, list[str]] = {
            x: y for x, y in zip(['brain_metaData', 'brain_df'], [self.brain_metaData.keys(), self.brain_df.keys()])
        }
        return str(dataKey_map)
        
    
    
    def visualize_images(self, n_images: Optional[int] = 4) -> 'plot':
        ...
    
    
    def diagnosis_plot(self, fig_size: Optional[tuple[int, int]] = (10, 6)) -> 'plot':
        ax = self.brain_df['diagnosis'].value_counts().plot(
            kind= 'bar',
            stacked= True,
            figsize= fig_size,
            color=["violet", "lightseagreen"]
        )
        ax.set_xticklabels(["Positive", "Negative"], rotation= 45, fontsize= 12)
        ax.set_ylabel('Total Images', fontsize = 12)
        ax.set_title("Distribution of data grouped by diagnosis",fontsize = 18, y=1.05)
        plt.show()