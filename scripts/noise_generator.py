import gradio as gr
import numpy as np
from modules import scripts, script_callbacks
from modules.shared import opts, cmd_opts, state
import json
import os
from PIL import Image
import random
from scripts.noise_algorithms import generate_simplex_noise, generate_marble_noise, generate_water_splash_noise, generate_colorful_voronoi
import piexif
import base64

class NoiseGenerator:
    def __init__(self):
        self.noise_params = {}
        self.last_generated_image = None
        self.seed = random.randint(0, 100000)

    def reset_seed(self):
        self.seed = random.randint(0, 100000)

    def generate_noise(self, width, height, brightness, color, noise_type, noise_scale, octaves, persistence, contrast, color_mode, color1, color2, color3, color4, gradient_angle, num_cells):
        self.reset_seed()

        # 色の値をリストから16進数文字列に変換
        color = self.ensure_hex_color(color)
        color1 = self.ensure_hex_color(color1)
        color2 = self.ensure_hex_color(color2)
        color3 = self.ensure_hex_color(color3)
        color4 = self.ensure_hex_color(color4)

        if noise_type == "White":
            noise_array = np.random.rand(height, width, 3)
        elif noise_type == "Simplex":
            noise_array = generate_simplex_noise(width, height, noise_scale, octaves, persistence, self.seed)
            noise_array = np.stack([noise_array] * 3, axis=-1)
        elif noise_type == "Marble":
            noise_array = generate_marble_noise(width, height, noise_scale, octaves, persistence, self.seed)
            noise_array = np.stack([noise_array] * 3, axis=-1)
        elif noise_type == "Water Splash":
            noise_array = generate_water_splash_noise(width, height, noise_scale, octaves, persistence, self.seed)
            noise_array = np.stack([noise_array] * 3, axis=-1)
        elif noise_type == "Colorful Voronoi":
            noise_array = generate_colorful_voronoi(width, height, num_cells, self.seed)
        
        # コントラスト調整（Colorful Voronoiの場合はスキップ）
        if noise_type != "Colorful Voronoi":
            noise_array = (noise_array - 0.5) * contrast + 0.5
            noise_array = np.clip(noise_array, 0, 1)

        # カラーモードに応じた処理（Colorful VoronoiとWhiteの場合はスキップ）
        if noise_type not in ["Colorful Voronoi", "White"]:
            if color_mode == "Grayscale":
                noise_array = np.mean(noise_array, axis=-1, keepdims=True)
                noise_array = np.repeat(noise_array, 3, axis=-1)
            elif color_mode == "Two Color":
                color1 = np.array(self.hex_to_rgb(color1)) / 255.0
                color2 = np.array(self.hex_to_rgb(color2)) / 255.0
                noise_array = color1[None, None, :] * noise_array + color2[None, None, :] * (1 - noise_array)
            elif color_mode == "Four Color":
                color1 = np.array(self.hex_to_rgb(color1)) / 255.0
                color2 = np.array(self.hex_to_rgb(color2)) / 255.0
                color3 = np.array(self.hex_to_rgb(color3)) / 255.0
                color4 = np.array(self.hex_to_rgb(color4)) / 255.0
                
                # if gradient_direction == "Horizontal":
                #     x = np.linspace(0, 1, width)
                #     y = np.linspace(0, 1, height)
                #     x, y = np.meshgrid(x, y)
                # elif gradient_direction == "Vertical":
                #     x = np.linspace(0, 1, height)
                #     y = np.linspace(0, 1, width)
                #     y, x = np.meshgrid(y, x)
                #     x = x.T
                #     y = y.T
                # elif gradient_direction == "Diagonal":
                angle_rad = np.radians(gradient_angle)
                x = np.linspace(0, 1, width)
                y = np.linspace(0, 1, height)
                x, y = np.meshgrid(x, y)
                rotated = x * np.cos(angle_rad) + y * np.sin(angle_rad)
                rotated = (rotated - rotated.min()) / (rotated.max() - rotated.min())
                x = y = rotated

                top = color1[None, None, :] * (1 - x[:, :, None]) + color2[None, None, :] * x[:, :, None]
                bottom = color3[None, None, :] * (1 - x[:, :, None]) + color4[None, None, :] * x[:, :, None]
                noise_array = top * noise_array + bottom * (1 - noise_array)

        # ベースカラーの適用（乗算ブレンド）
        if noise_type != "Colorful Voronoi":
            base_color = np.array(self.hex_to_rgb(color)) / 255.0
            noise_array = noise_array * base_color[None, None, :]
        
        # 明るさ調整
        noise_array = noise_array * brightness
        noise_array = np.clip(noise_array, 0, 1)
        
        noise_image = Image.fromarray((noise_array * 255).astype(np.uint8))
        self.last_generated_image = noise_image

        # パラメータをJSONに変換
        params = {
            'width': int(width),
            'height': int(height),
            'brightness': float(brightness),
            'color': color,
            'noise_type': noise_type,
            'noise_scale': float(noise_scale),
            'octaves': int(octaves),
            'persistence': float(persistence),
            'contrast': float(contrast),
            'color_mode': color_mode,
            'color1': color1,
            'color2': color2,
            'color3': color3,
            'color4': color4,
            'gradient_angle': float(gradient_angle),
            'num_cells': int(num_cells),
            'seed': int(self.seed)
        }

        # NumPy配列をリストに変換
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                params[key] = value.tolist()
            elif isinstance(value, np.generic):
                params[key] = value.item()
        json_params = json.dumps(params)

        # EXIFにJSONを埋め込む
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = json_params.encode('utf-8')
        exif_bytes = piexif.dump(exif_dict)
        noise_image.save('temp.png', 'PNG', exif=exif_bytes)
        
        return 'temp.png'

    def hex_to_rgb(self, hex_color):
        return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    def generate_random_colors(self):
        return [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(4)]

    def load_image_params(self, image_path):
        try:
            with Image.open(image_path) as img:
                exif_data = img._getexif()
                if exif_data:
                    image_description = exif_data.get(piexif.ImageIFD.ImageDescription)
                    if image_description:
                        if isinstance(image_description, bytes):
                            image_description = image_description.decode('utf-8')
                        params = json.loads(image_description)
                        
                        print("Loaded params:", params)  # デバッグ情報

                        # 色の値を確実に16進数文字列として扱う
                        for color_key in ['color', 'color1', 'color2', 'color3', 'color4']:
                            if color_key in params and params[color_key] is not None:
                                original_value = params[color_key]
                                params[color_key] = self.ensure_hex_color(params[color_key])
                                print(f"Color conversion for {color_key}: {original_value} -> {params[color_key]}")  # デバッグ情報
                        
                        return params
        except Exception as e:
            print(f"Error loading image parameters: {e}")
            print(f"Image path: {image_path}")
            if 'image_description' in locals():
                print(f"Image description type: {type(image_description)}")
                print(f"Image description content: {image_description}")
        return None

    def ensure_hex_color(self, color):
        print(f"ensure_hex_color input: {color}")  # デバッグ情報
        if isinstance(color, list):
            # リストの場合、RGBの値を16進数文字列に変換
            if all(isinstance(c, float) and 0 <= c <= 1 for c in color):
                # 浮動小数点数の場合（0-1の範囲）
                result = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
            else:
                # 整数の場合（0-255の範囲）
                result = f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"
        elif isinstance(color, str):
            # 文字列の場合、'#'で始まっていることを確認
            result = color if color.startswith('#') else f"#{color}"
        elif isinstance(color, int):
            # 整数の場合、16進数文字列に変換
            result = f"#{color:06x}"
        else:
            # その他の場合、デフォルト値を返す
            result = "#000000"
        print(f"ensure_hex_color output: {result}")  # デバッグ情報
        return result

noise_generator = NoiseGenerator()

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as noise_generator_tab:
        with gr.Row():
            with gr.Column():
                width = gr.Slider(minimum=64, maximum=512, step=64, label="Width", value=256)
                height = gr.Slider(minimum=64, maximum=512, step=64, label="Height", value=256)
                brightness = gr.Slider(minimum=0, maximum=2, step=0.1, label="Brightness", value=1.0)
                color = gr.ColorPicker(label="Base Color", value="#FFFFFF")
                noise_type = gr.Dropdown(["White", "Simplex", "Marble", "Water Splash", "Colorful Voronoi"], label="Noise Type", value="Simplex")
                noise_scale = gr.Slider(minimum=1, maximum=100, step=1, label="Noise Scale", value=10)
                octaves = gr.Slider(minimum=1, maximum=8, step=1, label="Octaves", value=4)
                persistence = gr.Slider(minimum=0, maximum=1, step=0.1, label="Persistence", value=0.5)
                contrast = gr.Slider(minimum=0.1, maximum=3, step=0.1, label="Contrast", value=1.0)
                color_mode = gr.Dropdown(["Grayscale", "Two Color", "Four Color"], label="Color Mode", value="Grayscale")
                with gr.Row(visible=False) as color_pickers:
                    color1 = gr.ColorPicker(label="Color 1", value="#FF0000")
                    color2 = gr.ColorPicker(label="Color 2", value="#0000FF")
                    color3 = gr.ColorPicker(label="Color 3", value="#00FF00")
                    color4 = gr.ColorPicker(label="Color 4", value="#FFFF00")
                with gr.Row(visible=False) as gradient_controls:
                    gradient_angle = gr.Slider(minimum=0, maximum=360, step=1, label="Gradient Angle (Diagonal only)", value=45)
                num_cells = gr.Slider(minimum=10, maximum=500, step=10, label="Number of Cells (Voronoi only)", value=50)
                generate_button = gr.Button("Generate Noise")
                randomize_colors_button = gr.Button("Randomize Colors")
                load_image_button = gr.UploadButton("Load Image", file_types=["image"])
            with gr.Column():
                result_image = gr.Image(label="Generated Noise")

        def update_color_pickers(color_mode):
            if color_mode == "Grayscale":
                return gr.update(visible=False), gr.update(visible=False)
            elif color_mode == "Two Color":
                return gr.update(visible=True), gr.update(visible=False)
            elif color_mode == "Four Color":
                return gr.update(visible=True), gr.update(visible=True)

        color_mode.change(
            fn=update_color_pickers,
            inputs=[color_mode],
            outputs=[color_pickers, gradient_controls],
        )

        def randomize_colors():
            new_colors = noise_generator.generate_random_colors()
            return new_colors[0], new_colors[1], new_colors[2], new_colors[3]

        randomize_colors_button.click(
            fn=randomize_colors,
            inputs=[],
            outputs=[color1, color2, color3, color4],
        )

        generate_button.click(
            fn=noise_generator.generate_noise,
            inputs=[width, height, brightness, color, noise_type, noise_scale, octaves, persistence, contrast, color_mode, color1, color2, color3, color4, gradient_angle, num_cells],
            outputs=[result_image],
        )

        def load_image(file):
            if file is not None:
                params = noise_generator.load_image_params(file.name)
                if params:
                    return [
                        gr.update(value=params['width']),
                        gr.update(value=params['height']),
                        gr.update(value=params['brightness']),
                        gr.update(value=params['color']),
                        gr.update(value=params['noise_type']),
                        gr.update(value=params['noise_scale']),
                        gr.update(value=params['octaves']),
                        gr.update(value=params['persistence']),
                        gr.update(value=params['contrast']),
                        gr.update(value=params['color_mode']),
                        gr.update(value=params['color1']),
                        gr.update(value=params['color2']),
                        gr.update(value=params['color3']),
                        gr.update(value=params['color4']),
                        gr.update(value=params['gradient_angle']),
                        gr.update(value=params['num_cells']),
                    ]
            return [gr.update()] * 17

        load_image_button.upload(
            fn=load_image,
            inputs=[load_image_button],
            outputs=[width, height, brightness, color, noise_type, noise_scale, octaves, persistence, contrast, color_mode, color1, color2, color3, color4, gradient_angle, num_cells],
        )

    return [(noise_generator_tab, "Noise Generator", "noise_generator_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)