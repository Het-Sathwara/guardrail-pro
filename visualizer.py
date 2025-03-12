import matplotlib.pyplot as plt
import numpy as np
from typing import List

def create_radar_chart(categories: List[str], values: List[float], thresholds: List[float]):
    values = values.copy()
    thresholds = thresholds.copy()
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    N = len(categories)
    
    angles += angles[:1]  # Close the loop
    
    values += values[:1]
    thresholds += thresholds[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label='Actual')
    ax.fill(angles, values, alpha=0.25)
    
    ax.plot(angles, thresholds, 'o-', linewidth=2, label='Threshold')
    
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    
    ax.set_ylim(0, 1)
    
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("Model Safety Radar", size=15, y=1.1)
    
    return fig

def create_bar_chart(categories: List[str], values: List[float], thresholds: List[float]):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_positions = np.arange(len(categories))
    bar_width = 0.35
    
    bars = ax.bar(
        bar_positions, 
        values, 
        bar_width, 
        label='Score',
        color=['green' if values[i] >= thresholds[i] else 'red' for i in range(len(values))]
    )
    
    ax.bar(
        bar_positions + bar_width, 
        thresholds, 
        bar_width, 
        label='Threshold',
        color='darkgray',
        alpha=0.7
    )
    
    ax.set_xlabel('Test Categories')
    ax.set_ylabel('Score')
    ax.set_title('Model Safety Test Scores')
    ax.set_xticks(bar_positions + bar_width / 2)
    ax.set_xticklabels(categories)
    ax.legend()
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    fig.tight_layout()
    
    return fig