Mix.install([
  {:jason, "~> 1.4.4"},
  {:nx, "~> 0.9.0"},
  {:tesla, "~> 1.12"}
])

defmodule EmbeddingClient do
  use Tesla

  def get_embedding(image_path) do
    url = "http://localhost:8800/predict"

    IO.puts("Sending request to #{url}")
    IO.puts("File: #{Path.basename(image_path)}")

    # Read the image file
    {:ok, image_data} = File.read(image_path)

    # Get the filename from the path
    filename = Path.basename(image_path)

    # Prepare the multipart form data
    mp =
      Tesla.Multipart.new()
      |> Tesla.Multipart.add_file_content(image_data, filename, name: "image")

    # Make the POST request
    case post(url, mp) do
      {:ok, %{status: 200, body: body}} ->
        # Assuming the response is JSON with an "embedding" key
        case Jason.decode(body) do
          {:ok, %{"embedding" => embedding}} ->
            {:ok, Nx.tensor(embedding)}

          {:error, _} ->
            {:error, "Failed to parse response JSON"}
        end

      {:ok, %{status: status}} ->
        {:error, "Request failed with status #{status}"}

      {:error, error} ->
        {:error, "Request failed: #{inspect(error)}"}
    end
  end
end

# Specify the path to your image file
image_path = "next/DSC01605.jpeg"

# Run the client and print the result
case EmbeddingClient.get_embedding(image_path) do
  {:ok, tensor} ->
    IO.puts("Successfully retrieved embedding:")
    # IO.inspect(tensor, limit: :infinity)
    IO.inspect(tensor)
    IO.puts("Tensor shape: #{inspect(Nx.shape(tensor))}")
    IO.puts("Tensor type: #{inspect(Nx.type(tensor))}")

  {:error, message} ->
    IO.puts("Error: #{message}")
end
