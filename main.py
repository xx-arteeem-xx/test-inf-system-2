import tkinter as tk
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

data = {
    'Страна': [
        'Бразилия', 'Россия', 'Индия', 'Китай', 'ЮАР',
        'Египет', 'Эфиопия', 'Иран', 'ОАЭ', 'Саудовская Аравия', 'Индонезия'
    ],
    'ВВП': [2179.41, 2173.84, 3912.87, 18743.8, 400.26, 389.06, 163.7, 436.91, 537.08, 1237.53, 1396.3],
    'ВВП_на_душу': [10280, 14889, 2696, 13303, 6253, 3338, 916.29, 4771, 49377, 35057, 4925],
    'Экспорт': [392.61, 476.43, 828.63, 3753.06, 127.48, 63.71, 2.86, 100, 1424, 360.91, 309.75],
    'Импорт': [381.76, 382.41, 919.2, 3219.34, 119.5, 90.36, 17.1, 117.18, 1388.3, 317.3, 284.7],
    'Торговый_баланс': [10.85, 94.02, 90.57, 533.72, 7.98, -26.65, -14.24, -17.18, 35.7, 43.61, 25.05],
}

FEATURES = ['ВВП', 'ВВП_на_душу', 'Экспорт', 'Импорт', 'Торговый_баланс']

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']


def _ru_plural(n, one, few, many):
    """Return correct Russian plural form for n."""
    if 11 <= n % 100 <= 19:
        return many
    r = n % 10
    if r == 1:
        return one
    if 2 <= r <= 4:
        return few
    return many


class ToplevelElbow(tk.Toplevel):
    def __init__(self, parent, X_scaled):
        super().__init__(parent)
        self.title('Метод локтя')
        self.resizable(False, False)

        k_range = list(range(2, 8))
        inertias = [
            KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_
            for k in k_range
        ]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(k_range, inertias, 'o-', color='steelblue', linewidth=2, markersize=6)
        for k, inertia in zip(k_range, inertias):
            ax.annotate(
                f'{inertia:.1f}',
                (k, inertia),
                textcoords='offset points',
                xytext=(0, 9),
                ha='center',
                fontsize=9,
            )
        ax.set_title('Зависимость инерции от количества кластеров')
        ax.set_xlabel('Количество кластеров (K)')
        ax.set_ylabel('Инерция')
        ax.set_xticks(k_range)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.protocol('WM_DELETE_WINDOW', self._on_close)

    def _on_close(self):
        self.destroy()


class ClusteringApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Многомерная кластеризация')

        self.df = pd.DataFrame(data)
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.df[FEATURES])
        self.pca = PCA(n_components=2)
        self.X_pca = self.pca.fit_transform(self.X_scaled)

        self._elbow_window = None

        self._build_ui()
        self._update_plot(2)

    def _build_ui(self):
        content = tk.Frame(self)
        content.pack(fill=tk.BOTH, expand=True)

        # Left: PCA scatter plot
        left = tk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right: cluster statistics panel
        right = tk.Frame(content, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        scrollbar = tk.Scrollbar(right)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text = tk.Text(
            right,
            wrap=tk.WORD,
            font=('Consolas', 9),
            state=tk.DISABLED,
            bg='#f5f5f5',
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=self.info_text.yview)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Bottom: label + slider + button
        bottom = tk.Frame(self, pady=6)
        bottom.pack(fill=tk.X)

        tk.Label(bottom, text='Количество кластеров:').pack(side=tk.LEFT, padx=(8, 2))

        self.slider = tk.Scale(
            bottom,
            from_=2,
            to=7,
            orient=tk.HORIZONTAL,
            showvalue=True,
            length=200,
            command=self._on_slider_change,
        )
        self.slider.set(2)
        self.slider.pack(side=tk.LEFT, padx=4)

        tk.Button(bottom, text='Метод локтя', command=self._show_elbow).pack(side=tk.LEFT, padx=8)

    def _on_slider_change(self, val):
        self._update_plot(int(float(val)))

    def _update_plot(self, k):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(self.X_scaled)

        self.ax.clear()

        for cluster_id in range(k):
            mask = labels == cluster_id
            self.ax.scatter(
                self.X_pca[mask, 0],
                self.X_pca[mask, 1],
                color=COLORS[cluster_id],
                s=80,
                label=f'Кластер {cluster_id + 1}',
                zorder=3,
            )

        for i, name in enumerate(self.df['Страна']):
            self.ax.annotate(
                name,
                (self.X_pca[i, 0], self.X_pca[i, 1]),
                fontsize=8,
                xytext=(4, 4),
                textcoords='offset points',
            )

        self.ax.set_title(f'PCA проекция (Кластеров: {k})')
        self.ax.set_xlabel('Главная компонента 1')
        self.ax.set_ylabel('Главная компонента 2')
        self.ax.legend(title='Кластеры', loc='upper right')
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

        self._update_info_panel(labels, k)

    def _update_info_panel(self, labels, k):
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)

        for cluster_id in range(k):
            mask = labels == cluster_id
            countries = self.df.loc[mask, 'Страна'].tolist()
            n = len(countries)
            mean_gdp = self.df.loc[mask, 'ВВП'].mean()
            mean_gdp_pc = self.df.loc[mask, 'ВВП_на_душу'].mean()
            mean_trade = self.df.loc[mask, 'Торговый_баланс'].mean()
            word = _ru_plural(n, 'страна', 'страны', 'стран')

            self.info_text.insert(tk.END, f'Кластер {cluster_id + 1} ({n} {word})\n')
            self.info_text.insert(tk.END, ', '.join(countries) + '\n')
            self.info_text.insert(tk.END, f'Средний ВВП: {mean_gdp:.1f} млрд $\n')
            self.info_text.insert(tk.END, f'Средний ВВП на душу: {mean_gdp_pc:.0f} $\n')
            self.info_text.insert(tk.END, f'Средний торговый баланс: {mean_trade:.1f} млрд $\n\n')

        self.info_text.configure(state=tk.DISABLED)

    def _show_elbow(self):
        if self._elbow_window is not None and self._elbow_window.winfo_exists():
            self._elbow_window.lift()
        else:
            self._elbow_window = ToplevelElbow(self, self.X_scaled)


if __name__ == '__main__':
    app = ClusteringApp()
    app.mainloop()
