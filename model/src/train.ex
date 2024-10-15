Mix.install([
  {:nx, "~> 0.7.0"},
  {:exla, "~> 0.7.0"},
  {:ortex, "~> 0.1.9"},
  {:bumblebee, "~> 0.5.3"},
  {:stb_image, "~> 0.6.9"},
  {:csv, "~> 2.4"}
])

defmodule ImageEmbedding do
  def process_image(image_path, model) do
    image =
      image_path
      |> StbImage.read_file!()
      |> StbImage.resize(224, 224)
      |> StbImage.to_nx()
      |> Nx.reshape({1, 3, 224, 224})
      |> Nx.as_type(:f32)

    {embeddings} = Ortex.run(model, image)
    embeddings
  end

  def save_embeddings(embeddings, filename) do
    binary_data = :erlang.term_to_binary(Nx.to_list(embeddings))
    File.write!(filename, binary_data)
  end

  def process_and_save(image_path, model, embedding_dir) do
    embedding_filename =
      Path.join(embedding_dir, "#{Path.basename(image_path, Path.extname(image_path))}.dat")

    embeddings =
      if File.exists?(embedding_filename) do
        IO.puts("Loading existing embeddings for #{image_path}")
        {:ok, binary_data} = File.read(embedding_filename)
        :erlang.binary_to_term(binary_data)
      else
        IO.puts("Processing image: #{image_path}")
        embeddings = process_image(image_path, model)
        save_embeddings(embeddings, embedding_filename)
        IO.puts("Saved embeddings for #{image_path} to #{embedding_filename}")
        embeddings
      end

    {Path.basename(image_path), embeddings}
  end
end

# Load the model
model = Ortex.load("resnet50_embeddings.onnx")

# Get all image files from the data/images folder
image_files = Path.wildcard("data/images/*.{jpg,jpeg}")

# Ensure the embeddings directory exists
embedding_dir = "data/embeddings_resnet50_elixir"
File.mkdir_p!(embedding_dir)

# Process images concurrently using Tasks
IO.puts("Number of schedulers: #{:erlang.system_info(:schedulers_online)}")
IO.puts("Number of logical processors: #{:erlang.system_info(:logical_processors_available)}")
IO.puts("Number of logical processors online: #{:erlang.system_info(:logical_processors_online)}")

# Process images concurrently using Tasks and collect results
embeddings_map =
  image_files
  |> Enum.map(fn image_path ->
    Task.async(fn -> ImageEmbedding.process_and_save(image_path, model, embedding_dir) end)
  end)
  |> Task.await_many(:infinity)
  |> Enum.into(%{})

IO.puts("All images processed successfully!")

defmodule ImageComparison do
  def normalize_embeddings(embeddings) do
    norms = Nx.sqrt(Nx.sum(Nx.pow(embeddings, 2), axes: [-1], keep_axes: true))
    Nx.divide(embeddings, norms)
  end

  def compare_new_image(new_image_path, model, embeddings_map) do
    # Process the new image
    new_embedding =
      ImageEmbedding.process_image(new_image_path, model)

    # Convert all embeddings to a single tensor
    # {names, embeddings} = Enum.unzip(embeddings_map)

    # existing_embeddings = Nx.stack(embeddings)

    existing_embeddings =
      embeddings_map
      |> Map.values()
      |> Enum.map(&Nx.tensor/1)
      |> Nx.stack()

    # Add new embedding to the stack
    # all_embeddings =
    #   Nx.concatenate([
    #     existing_embeddings,
    #     Nx.reshape(Nx.tensor(new_embedding), {1, 1, 2048, 1, 1})
    #   ])

    # Normalize all embeddings at once
    # normalized_embeddings = normalize_embeddings(all_embeddings)
    normalized_embeddings = normalize_embeddings(existing_embeddings)

    # Separate new embedding and existing embeddings
    {normalized_existing, normalized_new} = Nx.split(normalized_embeddings, -1)

    # Compute similarities
    similarities =
      Nx.vectorize(
        fn x -> Nx.sum(Nx.multiply(x, normalized_new)) end,
        normalized_existing
      )

    # Get indices of top 5 similar images
    top_indices = Nx.argsort(similarities, direction: :desc) |> Nx.slice([0], [5])

    # Get top 5 similarities and corresponding image names
    top_similarities = Nx.take(similarities, top_indices)
    names = Map.keys(embeddings_map)
    top_names = Enum.map(Nx.to_list(top_indices), &Enum.at(names, &1))

    Enum.zip(top_names, Nx.to_list(top_similarities))
  end
end

# Usage remains the same
new_image_path = "next/DSC01605.jpeg"

top_similar =
  ImageComparison.compare_new_image(new_image_path, model, embeddings_map)

# Print results
IO.puts("Top 5 similar images to #{new_image_path}:")

Enum.each(top_similar, fn {name, similarity} ->
  IO.puts("#{name}: #{Float.round(similarity, 3)}")
end)
