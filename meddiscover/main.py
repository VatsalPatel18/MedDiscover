from meddiscover.gradio_app import build_interface

def main():
    demo = build_interface()
    demo.launch(share=True)

if __name__ == "__main__":
    main()