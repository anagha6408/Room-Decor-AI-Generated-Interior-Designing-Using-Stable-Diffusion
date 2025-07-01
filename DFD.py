from graphviz import Digraph

# Create a Digraph object
dfd = Digraph("DFD_Level_0", format="png")

# Define nodes
dfd.attr('node', shape='rectangle', style='filled', color='lightblue')
dfd.node("User", "User")
dfd.node("Website", "Website Interface")
dfd.node("FlaskAPI", "Flask API")
dfd.node("StableDiffusion", "Stable Diffusion Model")
dfd.node("Output", "Generated Room Image")

# Define edges
dfd.attr('edge', arrowhead='open', arrowsize='1.2')
dfd.edge("User", "Website", "Uploads Image & Selects Style")
dfd.edge("Website", "FlaskAPI", "Sends Request with Image and Style")
dfd.edge("FlaskAPI", "StableDiffusion", "Processes Request and Runs Model")
dfd.edge("StableDiffusion", "FlaskAPI", "Returns Generated Image")
dfd.edge("FlaskAPI", "Website", "Sends Generated Image URL")
dfd.edge("Website", "User", "Displays Generated Image")

# Render DFD
output_file = dfd.render(filename="dfd_level_0")
print(f"DFD generated and saved as {output_file}")
