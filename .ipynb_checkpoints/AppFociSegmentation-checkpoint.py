import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
from skimage import morphology
from skimage import io, filters, measure, color, img_as_float
from skimage.morphology import disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import sklearn.neighbors as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os


def safe_mkdir(path):
    if not os.path.exists(path): os.makedirs(path)


def foci_stat(labels, image):
    # Extract properties like area, perimeter, etc.
    properties = measure.regionprops_table(labels, image, properties=('area', 'centroid', 'intensity_mean'))
    properties = pd.DataFrame(properties)
    properties.columns = ['Foci_area', 'Foci_centroid_0', 'Foci_centroid_1', 'Foci_intensity_mean']
    return properties


def assign_foci_to_nuclei(nuclei_stat_DF, foci_stat_DF):
    nuclei_coords = nuclei_stat_DF[['DAPI_centroid_0', 'DAPI_centroid_1']]
    nuclei_coords = np.array(nuclei_coords)
    foci_coords = foci_stat_DF[['Foci_centroid_0', 'Foci_centroid_1']]
    foci_coords = np.array(foci_coords)
    knn = nn.NearestNeighbors(n_neighbors=1, metric='euclidean')
    knn = knn.fit(nuclei_coords)
    distance, nearest_neuclei = knn.kneighbors(foci_coords)
    foci_stat_DF['nearest_nuclei_index'] = nearest_neuclei
    foci_stat_DF['nearest_nuclei_distance'] = distance
    nuclei_stat_DF['nuclei_index'] = nuclei_stat_DF.index
    result_DF = pd.merge(foci_stat_DF, nuclei_stat_DF, left_on='nearest_nuclei_index', right_on='nuclei_index', how='left')
    return result_DF


def stat_by_cell(final_stat, foci_area_cutoff=0., foci_intensity_cutoff=0.):
    final_stat = final_stat.query(f"Foci_area>{foci_area_cutoff} and Foci_intensity_mean>{foci_intensity_cutoff}").copy()
    final_stat.loc[:, 'Foci_intensity_sum'] = final_stat['Foci_intensity_mean'] * final_stat['Foci_area']
    stat_by_cell = final_stat.groupby('nuclei_index').agg(
        num_foci=('nuclei_index', 'count'),
        foci_intensity_sum=('Foci_intensity_sum', 'sum'),
        foci_area_sum=('Foci_area', 'sum'))
    # 计算foci_intensity_mean
    stat_by_cell.loc[:, 'foci_intensity_mean'] = stat_by_cell['foci_intensity_sum'] / stat_by_cell['foci_area_sum']
    stat_by_cell.loc[:, 'foci_area_mean'] = stat_by_cell['foci_area_sum'] / stat_by_cell['num_foci']
    # 去除重复行并选择需要的列
    stat_by_cell = stat_by_cell.drop_duplicates().reset_index()
    stat_by_cell = stat_by_cell[['nuclei_index', 'num_foci', 'foci_area_sum', 'foci_area_mean', 'foci_intensity_mean', 'foci_intensity_sum']]
    return stat_by_cell


def image_index(foci_image, 
                dapi_image, 
                foci_labels, 
                final_stat_DF, 
                nuclei_stat_DF,
                foci_intensity_cutoff=0., 
                foci_area_cutoff=0., 
                image_file='image_index.png', 
                show_image=False):
    image = np.stack([foci_image, np.zeros(dapi_image.shape), dapi_image], axis=2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    for region in measure.regionprops(foci_labels, foci_image):
        # take regions with large enough areas
        minr, minc, maxr, maxc = region.bbox
        if region.intensity_mean > foci_intensity_cutoff and region.area > foci_area_cutoff:
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='white', linewidth=.1)
            ax.add_patch(rect)
    final_stat_DF = final_stat_DF.query(f"Foci_area>{foci_area_cutoff} and Foci_intensity_mean>{foci_intensity_cutoff}").copy()
    for i in range(final_stat_DF.shape[0]):
        y,x,z=final_stat_DF[['Foci_centroid_0', 'Foci_centroid_1', 'nearest_nuclei_index']].iloc[i]
        ax.text(x, y, int(z), color='green', fontsize=3)
    for i in range(nuclei_stat_DF.shape[0]):
        y,x=nuclei_stat_DF[['DAPI_centroid_0', 'DAPI_centroid_1']].iloc[i]
        ax.text(x, y, i, color='orange', fontsize=5)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(image_file, dpi=600)
    if not show_image:
        plt.close()



class FociSegmentation:
    def __init__(self, root):
        self.root = root
        self.version = "0.0.1"
        self.author = "Jarning Gau"
        self.root.title(f'Foci segmentation v{self.version}\t By: {self.author}')

        # output
        self.out_path = "tmp"

        # 定义窗口大小和默认参数
        self.window_width = 2400
        self.window_height = 1600
        self.base_font_size = 10
        # 高通滤波
        self.default_sigma = 5
        self.default_channel = '0 - Red'
        self.default_binary_method = "global"

        # local threshold
        self.default_disk_r = 1.
        self.default_global_threshold = 0.5
        
        # segmentation
        self.default_segmentation_footprint = 3
        
        # 初始参数设置
        self.image = None
        self.channel_image = None
        self.binary_image = None
        self.sigma = self.default_sigma
        self.channel = self.default_channel
        self.disk_r = self.default_disk_r
        self.global_threshold = self.default_global_threshold
        self.segmentation_footprint = self.default_segmentation_footprint

        # 窗口定位居中
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int((screen_width - self.window_width) / 2)
        center_y = int((screen_height - self.window_height) / 2)
        self.root.geometry(f'{self.window_width}x{self.window_height}+{center_x}+{center_y}')

        # GUI布局
        self.layout_widgets()

    def layout_widgets(self):
        # ==== data loader ====
        # 加载图像按钮
        load_btn = tk.Button(self.root, text='Load Image', command=self.load_image, font=('Helvetica', int(self.base_font_size * 1.5)))
        load_btn.grid(row=0, column=0, padx=20, pady=20)

        # 通道选择下拉菜单
        channel_select_label = tk.Label(self.root, text="Select Channel:", font=('Helvetica', int(self.base_font_size * 1.5)))
        channel_select_label.grid(row=0, column=1, padx=10, pady=20, sticky='e')

        self.channel_var = tk.StringVar(value=self.default_channel)
        channel_select = ttk.Combobox(self.root, textvariable=self.channel_var,
                                      values=('0 - Red', '1 - Green', '2 - Blue'),
                                      state='readonly', font=('Helvetica', int(self.base_font_size * 1.5)))
        channel_select.current(0)  # 默认选择红色通道
        channel_select.grid(row=0, column=2, padx=10, pady=20)
        channel_select.bind('<<ComboboxSelected>>', self.set_channel)
        
        # 二值化方法选择下拉菜单
        bm_select_label = tk.Label(self.root, text="Select Binary Method:", font=('Helvetica', int(self.base_font_size * 1.5)))
        bm_select_label.grid(row=0, column=3, padx=10, pady=20, sticky='e')

        self.bm_var = tk.StringVar(value=self.default_binary_method)
        bm_select = ttk.Combobox(self.root, textvariable=self.bm_var, 
                                 values=('local', 'global'), state='readonly', font=('Helvetica', int(self.base_font_size * 1.5)))
        bm_select.current(1)
        bm_select.grid(row=0, column=4, padx=10, pady=20)
        bm_select.bind('<<ComboboxSelected>>', self.set_binary_method)

        # 重置按钮，将sigma和通道恢复默认值
        reset_btn = tk.Button(self.root, text='Reset Parameters', command=self.reset_parameters, font=('Helvetica', int(self.base_font_size * 1.5)))
        reset_btn.grid(row=0, column=5, padx=20, pady=20)

        # ==== 图像模糊 ====
        # 高斯模糊sigma滑块
        sigma_scale_label = tk.Label(self.root, text="Sigma for Gaussian Blur:", font=('Helvetica', int(self.base_font_size * 1.5)))
        sigma_scale_label.grid(row=1, column=0, padx=10, pady=5, sticky='e')

        self.sigma_scale = tk.Scale(self.root, from_=0.1, to=10, orient='horizontal',
                                    resolution=0.1, command=self.set_sigma, font=('Helvetica', int(self.base_font_size * 1.5)))
        self.sigma_scale.set(self.sigma)
        self.sigma_scale.grid(row=1, column=1, padx=10, pady=5, columnspan=3, sticky='we')

        # ==== 局部二值化 ====
        # disk_r滑块
        disk_r_scale_label = tk.Label(self.root, text="Radius for filter disk:", font=('Helvetica', int(self.base_font_size * 1.5)))
        disk_r_scale_label.grid(row=2, column=0, padx=10, pady=5, sticky='e')
        
        self.disk_r_scale = tk.Scale(self.root, from_=1, to=10, orient='horizontal',
                                        resolution=1, command=self.set_disk_r, font=('Helvetica', int(self.base_font_size * 1.5)))
        self.disk_r_scale.set(self.disk_r)
        self.disk_r_scale.grid(row=2, column=1, padx=10, pady=5, columnspan=3, sticky='we')
        
        # global threshold滑块
        global_threshold_label = tk.Label(self.root, text="Global threshold:", font=('Helvetica', int(self.base_font_size * 1.5)))
        global_threshold_label.grid(row=3, column=0, padx=10, pady=5, sticky='e')

        self.global_threshold_scale = tk.Scale(self.root, from_=0, to=1, orient='horizontal',
                                        resolution=0.02, command=self.set_global_threshold, font=('Helvetica', int(self.base_font_size * 1.5)))
        self.global_threshold_scale.set(self.global_threshold)
        self.global_threshold_scale.grid(row=3, column=1, padx=10, pady=5, columnspan=3, sticky='we')

        # segmentation
        segmentation_label = tk.Label(self.root, text="footprint for segmentation: ", font=('Helvetica', int(self.base_font_size * 1.5)))
        segmentation_label.grid(row=4, column=0, padx=10, pady=5, sticky='e')

        self.segmentation_footprint_scale = tk.Scale(self.root, from_=1, to=31, orient='horizontal', 
                                                     resolution=2, command=self.set_segmentation_footprint, font=('Helvetica', int(self.base_font_size * 1.5)))
        self.segmentation_footprint_scale.set(self.segmentation_footprint)
        self.segmentation_footprint_scale.grid(row=4, column=1, padx=10, pady=5, columnspan=3, sticky='we')

        segmentation_btn = tk.Button(self.root, text='Segmentation', command=self.foci_segmentation, font=('Helvetica', int(self.base_font_size * 1.5)))
        segmentation_btn.grid(row=4, column=4, padx=20, pady=20)

        # save results
        save_btn = tk.Button(self.root, text='Save', command=self.save, font=('Helvetica', int(self.base_font_size * 1.5)))
        save_btn.grid(row=4, column=5, padx=20, pady=20)


        # 画布-高斯模糊
        self.blurred_image_canvas = tk.Canvas(self.root, width=self.window_width // 2 - 40, height=self.window_height - 160)
        self.blurred_image_canvas.grid(row=5, column=0, padx=20, pady=5, columnspan=4)

        # 画布-局部二值化
        self.binary_image_canvas = tk.Canvas(self.root, width=self.window_width // 2 - 40, height=self.window_height - 160)
        self.binary_image_canvas.grid(row=5, column=4, padx=20, pady=5, columnspan=4)

    def reset_parameters(self):
        # 重置参数到默认值
        self.sigma_scale.set(self.default_sigma)
        self.channel_var.set(self.default_channel)
        self.bm_var.set(self.default_binary_method)
        self.disk_r_scale.set(self.default_disk_r)
        self.segmentation_footprint_scale.set(self.default_segmentation_footprint)
        self.update_layout()

    def show_images(self):
        if self.image is not None:
            # 获取用户选定的通道
            channel_map = {
                '0 - Red': 0,
                '1 - Green': 1,
                '2 - Blue': 2
            }
            selected_channel = channel_map[self.channel_var.get()]
            channel_image = img_as_float(self.image[:, :, selected_channel])
            self.channel_image = channel_image

            # 应用高斯模糊
            blurred = filters.gaussian(channel_image, sigma=self.sigma)
            # 高通滤波
            highpass = channel_image - blurred
            highpass = highpass - np.min(highpass)
            highpass = highpass / np.max(highpass)
            highpass_denoise = filters.median(highpass, disk(self.disk_r))  # 使用半径为1的圆盘型结构元素

            # 更新模糊图像画布
            blurred_image_pil = Image.fromarray((channel_image * 255).astype('uint8'))
            self.blurred_photo = ImageTk.PhotoImage(blurred_image_pil)
            self.blurred_image_canvas.create_image(0, 0, anchor='nw', image=self.blurred_photo)
            
            # 局部二值化
            if self.bm_var.get() == "local":
	            threshold_value = filters.threshold_otsu(highpass_denoise)
	            self.binary_image = highpass_denoise > threshold_value

            # 全局二值化
            if self.bm_var.get() == "global":
            	self.binary_image = highpass_denoise > self.global_threshold

            # 更新局部二值化画布
            if self.binary_image is not None:
	            binary_image_pil = Image.fromarray((self.binary_image * 255).astype('uint8'))
	            self.binary_photo = ImageTk.PhotoImage(binary_image_pil)
	            self.binary_image_canvas.create_image(0, 0, anchor='nw', image=self.binary_photo)
            
    
    def load_image(self):
        filepath = filedialog.askopenfilename()
        if not filepath:
            return
        
        self.image = io.imread(filepath)
        self.image_path = filepath
        self.show_images()

    def update_layout(self):
        self.show_images()

    def foci_segmentation(self):
        binary_image = self.binary_image
        distance = ndi.distance_transform_edt(binary_image)
        distance = filters.gaussian(distance, sigma=1)
        footprint = (self.segmentation_footprint, self.segmentation_footprint)
        coords = peak_local_max(distance, footprint=np.ones(footprint), labels=binary_image)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=binary_image)
        self.labels = labels
        for region in measure.regionprops(labels):
            # minr, minc, maxr, maxc = region.bbox # left-top (minr, minc), right-bottom (maxr, maxc)
            y1,x1, y2,x2 = region.bbox
            self.blurred_image_canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)


    def save(self):
        dir_name, basename = os.path.split(self.image_path)
        out_dir = os.path.join(dir_name, self.out_path)
        safe_mkdir(out_dir)
        foci_stat_DF = foci_stat(self.labels, self.channel_image)
        foci_stat_DF.to_csv(os.path.join(out_dir, "foci_stat.csv"))
        nuclei_stat_DF = pd.read_csv(os.path.join(out_dir, "nucleus_stat.csv"), index_col=0)
        merged_stat_DF = assign_foci_to_nuclei(nuclei_stat_DF, foci_stat_DF)
        merged_stat_DF.to_csv(os.path.join(out_dir, "merged_stat.csv"))
        per_cell_stat_DF = stat_by_cell(merged_stat_DF)
        per_cell_stat_DF.to_csv(os.path.join(dir_name, "per_cell_stat.csv"), index=False)
        # np.save(os.path.join(out_dir, "foci_image.npy"), self.channel_image)
        # index image
        foci_image = self.channel_image
        dapi_image = np.load(os.path.join(out_dir, "nucleus_image.npy"))
        foci_labels = self.labels
        image_index(foci_image, dapi_image, foci_labels, merged_stat_DF, nuclei_stat_DF,
                    image_file=os.path.join(dir_name, "indexed_image.png"))
        showinfo(title='Task status', message="Successfully saved!")


    def set_channel(self, event):
        self.update_layout()

    def set_binary_method(self, event):
    	self.update_layout()

    def set_sigma(self, val):
        self.sigma = float(val)
        self.update_layout()

    def set_disk_r(self, val):
    	self.disk_r = float(val)
    	self.update_layout()

    def set_global_threshold(self, val):
    	self.global_threshold = float(val)
    	self.update_layout()

    def set_segmentation_footprint(self, val):
        self.segmentation_footprint = int(val)
        self.update_layout()

    def run(self):
        self.root.mainloop()
