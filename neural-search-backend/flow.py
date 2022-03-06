from jina import Flow

flow = (
    Flow(port=8080)
    .add(uses="jinahub+docker://CLIPImageEncoder/v0.4")
    .add(
        uses="jinahub+docker://SimpleIndexer/v0.15",
    )
)

if __name__ == "__main__":
    flow.to_k8s_yaml("flow_k8s_configuration", k8s_namespace="findmybike")
   
