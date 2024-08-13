from provider_dependency.text_embedding import *
from typing import List, Dict, Optional


class TongyiTextEmbeddingModel(BaseTextEmbeddingModel):
    async def embed_text(
        self,
        provider_model_id: str,
        input: List[str],
        credentials: ProviderCredentials,
        configs: TextEmbeddingModelConfiguration,
        input_type: Optional[TextEmbeddingInputType] = None,
        proxy: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
    ) -> TextEmbeddingResult:

        api_url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

        headers = {
            "Authorization": f"Bearer {credentials.TONGYI_API_KEY}",
            "Content-Type": "application/json",
        }

        texts = {"texts": input}

        payload = {"model": provider_model_id, "input": texts}

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload, proxy=CONFIG.PROXY) as response:
                await self.handle_response(response)
                response_json = await response.json()
                return TextEmbeddingResult(
                    data=[
                        TextEmbeddingOutput(embedding=output["embedding"], index=output["text_index"])
                        for output in response_json["output"]["embeddings"]
                    ],
                )
