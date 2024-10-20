Mix.install([
  {:jason, "~> 1.4.4"},
  {:nx, "~> 0.9.0"},
  {:tesla, "~> 1.12"},
  {:hackney, "~> 1.20.1"}
])

defmodule PredictionClient do
  use Tesla

  plug Tesla.Middleware.Logger

  adapter Tesla.Adapter.Hackney, [
    transport_opts: [
      inet6: true
    ]
  ]

  def get_similar_images(image_path) do
    url = "http://[::1]:8800/predict"

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
      # |> Tesla.Multipart.add_file_content(image_data, filename, name: "image", detect_content_type: true)

    # Make the POST request
    case post(url, mp) do
      {:ok, %{status: 200, body: body}} ->
        case Jason.decode(body) do
          {:ok, %{"similar_images" => similar_images}} ->
            {:ok, similar_images}

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
case PredictionClient.get_similar_images(image_path) do
  {:ok, similar_images} ->
    IO.puts("Successfully retrieved similar images:")

    Enum.each(similar_images, fn image ->
      IO.puts("Rank: #{image["rank"]}")
      IO.puts("Filename: #{image["filename"]}")
      IO.puts("Similarity: #{image["similarity_percentage"]}%")
      IO.puts("Base64: #{String.slice(image["image_base64"], 0, 10)}")
      IO.puts("---")
    end)

  {:error, message} ->
    IO.puts("Error: #{message}")
end
