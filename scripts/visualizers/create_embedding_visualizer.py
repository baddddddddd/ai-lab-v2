from embedding_visualizer import EmbeddingVisualizer
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
embedding = model.transformer.wte
vocab = tokenizer.get_vocab()

inv_vocab = {v: k for k, v in vocab.items()}

visualizer = EmbeddingVisualizer(n_components=2, n_neighbors=15)
visualizer.load_embeddings(embedding, inv_vocab)
visualizer.fit_umap()
fig = visualizer.create_interactive_plot(sample_size=5000, show_clusters=True)
fig.write_html("my_embedding_plot.html")
visualizer.show_plot(fig, auto_open=True)
