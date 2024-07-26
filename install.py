import launch

if not launch.is_installed("scipy"):
    launch.run_pip("install scipy", "requirements for MagicPrompt")

if not launch.is_installed("piexif"):
    launch.run_pip("install piexif", "requirements for MagicPrompt")