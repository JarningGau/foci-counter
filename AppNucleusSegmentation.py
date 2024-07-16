import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
from skimage import morphology
from skimage import io, filters, measure, color, img_as_float
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import numpy as np
import pandas as pd
import os


def safe_mkdir(path):
    if not os.path.exists(path): os.makedirs(path)

def nuclei_stat(labels, dapi_image):
    # Extract properties like area, perimeter, etc.
    properties = measure.regionprops_table(labels, dapi_image, properties=('area', 'centroid', 'intensity_mean'))
    prop_dapi = pd.DataFrame(properties)
    prop_dapi.columns = ['DAPI_area', 'DAPI_centroid_0', 'DAPI_centroid_1', 'DAPI_intensity_mean']
    return prop_dapi


class NucleusSegmentation:
    def __init__(self, root):
        self.root = root
        self.version = "0.0.1"
        self.author = "Jarning Gau"
        self.root.title(f'Nucleus segmentation v{self.version}\t By: {self.author}')

        # output
        self.out_path = "tmp"

        # 定义窗口大小和默认参数
        self.window_width = 2400 # 1280
        self.window_height = 1600
        self.base_font_size = 10
        # 高斯模糊
        self.default_sigma = 3
        self.default_channel = '2 - Blue'
        # local threshold
        self.default_sauvola_k = 0.1
        self.default_sauvola_window_size = 21
        self.default_remove_small_region_min_size = 500
        # segmentation
        self.default_segmentation_footprint = 51
        
        
        # 初始参数设置
        self.sigma = self.default_sigma
        self.channel = self.default_channel
        self.image = None
        self.channel_image = None
        self.cleaned_image = None
        self.sauvola_k = self.default_sauvola_k
        self.sauvola_window_size = self.default_sauvola_window_size
        self.remove_small_region_min_size = self.default_remove_small_region_min_size
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

        # 重置按钮，将sigma和通道恢复默认值
        reset_btn = tk.Button(self.root, text='Reset Parameters', command=self.reset_parameters, font=('Helvetica', int(self.base_font_size * 1.5)))
        reset_btn.grid(row=0, column=3, padx=20, pady=20)

        # 通道选择下拉菜单
        channel_select_label = tk.Label(self.root, text="Select Channel:", font=('Helvetica', int(self.base_font_size * 1.5)))
        channel_select_label.grid(row=0, column=1, padx=10, pady=20, sticky='e')

        self.channel_var = tk.StringVar(value=self.default_channel)
        channel_select = ttk.Combobox(self.root, textvariable=self.channel_var,
                                      values=('0 - Red', '1 - Green', '2 - Blue'),
                                      state='readonly', font=('Helvetica', int(self.base_font_size * 1.5)))
        channel_select.current(2)  # 默认选择蓝色通道
        channel_select.grid(row=0, column=2, padx=10, pady=20)
        channel_select.bind('<<ComboboxSelected>>', self.set_channel)
        
        # ==== 图像模糊 ====
        # 高斯模糊sigma滑块
        sigma_scale_label = tk.Label(self.root, text="Sigma for Gaussian Blur:", font=('Helvetica', int(self.base_font_size * 1.5)))
        sigma_scale_label.grid(row=1, column=0, padx=10, pady=5, sticky='e')

        self.sigma_scale = tk.Scale(self.root, from_=0.1, to=10, orient='horizontal',
                                    resolution=0.1, command=self.set_sigma, font=('Helvetica', int(self.base_font_size * 1.5)))
        self.sigma_scale.set(self.sigma)
        self.sigma_scale.grid(row=1, column=1, padx=10, pady=5, columnspan=3, sticky='we')

        # ==== 局部二值化 ====
        # local threshold滑块
        # sauvola_k
        sauvola_k_scale_label = tk.Label(self.root, text="K for local sauvola:", font=('Helvetica', int(self.base_font_size * 1.5)))
        sauvola_k_scale_label.grid(row=2, column=0, padx=10, pady=5, sticky='e')
        
        self.sauvola_k_scale = tk.Scale(self.root, from_=0., to=1., orient='horizontal',
                                        resolution=0.01, command=self.set_sauvola_k, font=('Helvetica', int(self.base_font_size * 1.5)))
        self.sauvola_k_scale.set(self.sauvola_k)
        self.sauvola_k_scale.grid(row=2, column=1, padx=10, pady=5, columnspan=3, sticky='we')

        # sauvola_window_size
        sauvola_window_size_label = tk.Label(self.root, text="window size for local sauvola: ", font=('Helvetica', int(self.base_font_size * 1.5)))
        sauvola_window_size_label.grid(row=3, column=0, padx=10, pady=5, sticky='e')
        
        self.sauvola_window_size_scale = tk.Scale(self.root, from_=11, to=501, orient='horizontal',
                                                  resolution=10, command=self.set_sauvola_window_size, font=('Helvetica', int(self.base_font_size * 1.5)))
        self.sauvola_window_size_scale.set(self.sauvola_window_size)
        self.sauvola_window_size_scale.grid(row=3, column=1, padx=10, pady=5, columnspan=3, sticky='we')

        # remove small region: min size
        rsr_ms_label = tk.Label(self.root, text="min size for remove small regions: ", font=('Helvetica', int(self.base_font_size * 1.5)))
        rsr_ms_label.grid(row=4, column=0, padx=10, pady=5, sticky='e')

        self.rsr_ms_scale = tk.Scale(self.root, from_=100, to=10000, orient='horizontal', 
                                     resolution=100, command=self.set_rsr_ms, font=('Helvetica', int(self.base_font_size * 1.5)))
        self.rsr_ms_scale.set(self.remove_small_region_min_size)
        self.rsr_ms_scale.grid(row=4, column=1, padx=10, pady=5, columnspan=3, sticky='we')

        # segmentation
        segmentation_label = tk.Label(self.root, text="footprint for segmentation: ", font=('Helvetica', int(self.base_font_size * 1.5)))
        segmentation_label.grid(row=5, column=0, padx=10, pady=5, sticky='e')

        self.segmentation_footprint_scale = tk.Scale(self.root, from_=11, to=501, orient='horizontal', 
                                                     resolution=10, command=self.set_segmentation_footprint, font=('Helvetica', int(self.base_font_size * 1.5)))
        self.segmentation_footprint_scale.set(self.segmentation_footprint)
        self.segmentation_footprint_scale.grid(row=5, column=1, padx=10, pady=5, columnspan=3, sticky='we')

        segmentation_btn = tk.Button(self.root, text='Segmentation', command=self.nuclei_segmentation, font=('Helvetica', int(self.base_font_size * 1.5)))
        segmentation_btn.grid(row=5, column=4, padx=20, pady=20)

        # save results
        save_btn = tk.Button(self.root, text='Save', command=self.save, font=('Helvetica', int(self.base_font_size * 1.5)))
        save_btn.grid(row=5, column=5, padx=20, pady=20)


        # 画布初始化
        # self.raw_image_canvas = tk.Canvas(self.root, width=self.window_width // 2 - 40, height=self.window_height - 160)
        # self.raw_image_canvas.grid(row=4, column=0, padx=20, pady=5, columnspan=4)

        # 画布-高斯模糊
        self.blurred_image_canvas = tk.Canvas(self.root, width=self.window_width // 2 - 40, height=self.window_height - 160)
        self.blurred_image_canvas.grid(row=6, column=0, padx=20, pady=5, columnspan=4)

        # 画布-局部二值化
        self.binary_image_canvas = tk.Canvas(self.root, width=self.window_width // 2 - 40, height=self.window_height - 160)
        self.binary_image_canvas.grid(row=6, column=4, padx=20, pady=5, columnspan=4)

    def reset_parameters(self):
        # 重置参数到默认值
        self.sigma_scale.set(self.default_sigma)
        self.channel_var.set(self.default_channel)
        self.sauvola_k_scale.set(self.default_sauvola_k)
        self.sauvola_window_size_scale.set(self.default_sauvola_window_size)
        self.rsr_ms_scale.set(self.default_remove_small_region_min_size)
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
            # 更新原始图像画布
            # raw_image_pil = Image.fromarray((channel_image * 255).astype('uint8'))
            # self.raw_photo = ImageTk.PhotoImage(raw_image_pil)
            # self.raw_image_canvas.create_image(0, 0, anchor='nw', image=self.raw_photo)

            # 应用高斯模糊
            blurred = filters.gaussian(channel_image, sigma=self.sigma)

            # 更新模糊图像画布
            blurred_image_pil = Image.fromarray((blurred * 255).astype('uint8'))
            self.blurred_photo = ImageTk.PhotoImage(blurred_image_pil)
            self.blurred_image_canvas.create_image(0, 0, anchor='nw', image=self.blurred_photo)
            
            # 局部二值化
            window_size = self.sauvola_window_size
            k = self.sauvola_k
            local_threshold = filters.threshold_sauvola(blurred, window_size=window_size, k=k)
            binary_image = blurred > local_threshold
            fill_image = ndi.binary_fill_holes(binary_image)
            min_size = self.remove_small_region_min_size
            self.cleaned_image = morphology.remove_small_objects(fill_image, min_size=min_size, connectivity=1)
            # 更新局部二值化画布
            binary_image_pil = Image.fromarray((self.cleaned_image * 255).astype('uint8'))
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

    def nuclei_segmentation(self):
        cleaned_image = self.cleaned_image
        distance = ndi.distance_transform_edt(cleaned_image)
        distance = filters.gaussian(distance, sigma=3)
        footprint = (self.segmentation_footprint, self.segmentation_footprint)
        coords = peak_local_max(distance, footprint=np.ones(footprint), labels=cleaned_image)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=cleaned_image)
        self.labels = labels
        for region in measure.regionprops(labels):
            # minr, minc, maxr, maxc = region.bbox # left-top (minr, minc), right-bottom (maxr, maxc)
            y1,x1, y2,x2 = region.bbox
            self.blurred_image_canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)


    def save(self):
        dir_name, basename = os.path.split(self.image_path)
        out_dir = os.path.join(dir_name, self.out_path)
        safe_mkdir(out_dir)
        properties_DF = nuclei_stat(self.labels, self.channel_image)
        properties_DF.to_csv(os.path.join(out_dir, "nucleus_stat.csv"))
        np.save(os.path.join(out_dir, "nucleus_image.npy"), self.channel_image)
        showinfo(title='Task status', message="Successfully saved!")

    def set_channel(self, event):
        self.update_layout()

    def set_sigma(self, val):
        self.sigma = float(val)
        self.update_layout()

    def set_sauvola_k(self, val):
        self.sauvola_k = float(val)
        self.update_layout()

    def set_sauvola_window_size(self, val):
        self.sauvola_window_size = int(val)
        self.update_layout()

    # remove small region: min size
    def set_rsr_ms(self, val):
        self.remove_small_region_min_size = int(val)
        self.update_layout()

    def set_segmentation_footprint(self, val):
        self.segmentation_footprint = int(val)
        self.update_layout()
        if self.cleaned_image is not None:
            self.nuclei_segmentation()

    def run(self):
        self.root.mainloop()
