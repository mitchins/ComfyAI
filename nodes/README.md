# Available Nodes

This folder contains all custom nodes shipped with ComfyAI. They are automatically registered when the environment variable `UNIT_TEST_MODE` is not set.

| Node | Description |
|------|-------------|
| `VLLMTextQuery` | Send a text prompt to an OpenAI compatible endpoint and return the raw text reply together with a boolean interpretation. |
| `VLLMImageQuery` | Like `VLLMTextQuery` but also sends one image. |
| `VLLMDualImageQuery` | Sends two images along with the prompt, useful for comparison tasks. |
| `ConditionalSaveImage` | Saves incoming images to the ComfyUI output directory only when a boolean input is `True`. |
| `CompareFacesNode` | Uses the face comparison API in `apps/face_api` to determine whether two images depict the same person. |

Every node exposes standard `INPUT_TYPES` and `RETURN_TYPES` as expected by ComfyUI.
