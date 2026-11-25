package io.course.triage.config;

import java.time.Duration;

import com.agentx.ai.core.chatmodels.DeepSeekV4ChatModel;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.deepseek.DeepSeekChatOptions;
import org.springframework.ai.deepseek.api.DeepSeekApi;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.retry.policy.NeverRetryPolicy;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;

@Configuration
class ModelConfig {
    @Bean
    ChatModel triageModel(
            @Value("${triage.api-key}") String key,
            @Value("${triage.model-url}") String baseUrl,
            @Value("${triage.model}") String model) {
        if (key.isBlank()) {
            throw new IllegalStateException("Set DEEPSEEK_API_KEY before starting the backend");
        }
        var http = HttpClient.create().responseTimeout(Duration.ofSeconds(30));
        var api = DeepSeekApi.builder()
                .baseUrl(baseUrl)
                .apiKey(key)
                .webClientBuilder(WebClient.builder().clientConnector(new ReactorClientHttpConnector(http)))
                .build();
        var retry = new RetryTemplate();
        retry.setRetryPolicy(new NeverRetryPolicy());
        return DeepSeekV4ChatModel.builder()
                .deepSeekApi(api)
                .retryTemplate(retry)
                .defaultOptions(DeepSeekChatOptions.builder()
                        .model(model)
                        .temperature(0.0)
                        .internalToolExecutionEnabled(false)
                        .build())
                .build();
    }
}
