package io.course.triage.ml;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

@Component
public class MlClient {
    private final RestClient http;
    private final ObjectMapper json;

    public MlClient(@Value("${triage.ml-url}") String baseUrl, ObjectMapper json) {
        this.json = json;
        var factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(Duration.ofSeconds(2));
        factory.setReadTimeout(Duration.ofSeconds(5));
        http = RestClient.builder().baseUrl(baseUrl).requestFactory(factory).build();
    }

    public Prediction classify(String text) {
        return post("/classify", Map.of("text", text), Prediction.class);
    }

    public Evidence retrieve(String text, int topK) {
        return post("/retrieve", Map.of("text", text, "topK", topK), Evidence.class);
    }

    private <T> T post(String route, Map<String, ?> request, Class<T> type) {
        try {
            byte[] body = json.writeValueAsBytes(request);
            return Objects.requireNonNull(http.post().uri(route)
                    .contentType(MediaType.APPLICATION_JSON).contentLength(body.length)
                    .body(body).retrieve().body(type));
        } catch (JsonProcessingException | RestClientException | NullPointerException ex) {
            throw new IllegalStateException("Local ML service could not return " + route, ex);
        }
    }

    public record Prediction(String label, double confidence, double margin, List<String> features) {}
    public record Evidence(String label, double agreement, Map<String, Double> distribution) {}
}
