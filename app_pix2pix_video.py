import gradio as gr
from model import Model
import os
on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"


def create_demo(model: Model):
    examples = [
        ['__assets__/pix2pix_video_2fps/camel.mp4',
            'make it Van Gogh Starry Night style', 512, 0, 1.0],
        ['__assets__/pix2pix_video_2fps/mini-cooper.mp4',
            'make it Picasso style', 512, 0, 1.5],
        ['__assets__/pix2pix_video_2fps/snowboard.mp4',
            'replace man with robot', 512, 0, 1.0],
        ['__assets__/pix2pix_video_2fps/white-swan.mp4',
            'replace swan with mallard', 512, 0, 1.5],
        # ['__assets__/pix2pix_video_2fps/boat.mp4',
        #     'add city skyline in the background', 512, 0, 1.5],
        # ['__assets__/pix2pix_video_2fps/ballet.mp4',
        #     'make her a golden sculpture', 512, 0, 1.0],
    ]
    with gr.Blocks() as demo:

        with gr.Row():
            with gr.Column():
                input_image = gr.Video(label="Input Video", source='upload',
                                       type='numpy', format="mp4", visible=True).style(height="auto")
            with gr.Column():
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')
                with gr.Accordion('Advanced options', open=False):
                    image_resolution = gr.Slider(label='Image Resolution',
                                                 minimum=256,
                                                 maximum=1024,
                                                 value=512,
                                                 step=64)
                    seed = gr.Slider(label='Seed',
                                     minimum=0,
                                     maximum=65536,
                                     value=0,
                                     step=1)
                    image_guidance = gr.Slider(label='Image guidance scale',
                                               minimum=0.5,
                                               maximum=2,
                                               value=1.0,
                                               step=0.1)
                    start_t = gr.Slider(label='Starting time in seconds',
                                        minimum=0,
                                        maximum=10,
                                        value=0,
                                        step=1)
                    end_t = gr.Slider(label='End time in seconds (-1 corresponds to uploaded video duration)',
                                      minimum=0,
                                      maximum=10,
                                      value=-1,
                                      step=1)
                    out_fps = gr.Slider(label='Output video fps (-1 corresponds to uploaded video fps)',
                                        minimum=1,
                                        maximum=30,
                                        value=-1,
                                        step=1)
                    chunk_size = gr.Slider(
                        label="Chunk size", minimum=2, maximum=8, value=8, step=1, visible=not on_huggingspace)
            with gr.Column():
                result = gr.Video(label='Output', show_label=True)
        inputs = [
            input_image,
            prompt,
            image_resolution,
            seed,
            image_guidance,
            start_t,
            end_t,
            out_fps,
            chunk_size,
            None
        ]

        gr.Examples(examples=examples,
                    inputs=inputs,
                    outputs=result,
                    fn=model.process_pix2pix,
                    cache_examples=on_huggingspace,
                    run_on_click=False,
                    )

        run_button.click(fn=model.process_pix2pix,
                         inputs=inputs,
                         outputs=result)
    return demo
