package io.course.triage.agent;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.course.triage.ml.MlClient;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;

final class TriageTools {
    static final String SKILL_NAME = "news-topic-classification";
    private final ObjectMapper json;
    private final MlClient ml;

    TriageTools(ObjectMapper json, MlClient ml) {
        this.json = json;
        this.ml = ml;
    }

    List<ToolCallback> bind(TriageSession session, String text, ToolCallback skill) {
        var tools = new ArrayList<ToolCallback>();
        if (session.useSkill) {
            tools.add(new ToolCallback() {
                @Override
                public ToolDefinition getToolDefinition() {
                    return skill.getToolDefinition();
                }

                @Override
                public String call(String input) {
                    try {
                        requireOpen(session);
                        if (!SKILL_NAME.equals(json.readTree(input).path("command").asText())) {
                            throw new IllegalArgumentException("Unknown skill");
                        }
                        String loaded = json.readValue(skill.call(input), String.class);
                        int start = loaded.indexOf("\n\n");
                        if (start < 0 || !loaded.contains("## Codebook")) {
                            throw new IllegalStateException("Skill content unavailable");
                        }
                        session.skillLoaded = true;
                        session.trace.add(new TriageResult.ToolTrace(
                                getToolDefinition().name(), true, Map.of("name", SKILL_NAME)));
                        return loaded.substring(start + 2);
                    } catch (JsonProcessingException | RuntimeException ex) {
                        return failure(session, getToolDefinition().name(), "Could not load the named skill.");
                    }
                }
            });
        }
        tools.add(tool(session, "classify_topic", "Run the local TF-IDF + LinearSVC classifier.",
                "{\"type\":\"object\",\"properties\":{},\"additionalProperties\":false}",
                args -> {
                    session.prediction = ml.classify(text);
                    session.evidence = null;
                    return session.prediction;
                }));
        tools.add(tool(session, "retrieve_evidence",
                "Compare local TF-IDF neighbors after classification. top_k ranges from 3 to 15.",
                """
                {"type":"object","properties":{"top_k":{"type":"integer","minimum":3,"maximum":15}},
                 "required":["top_k"],"additionalProperties":false}
                """,
                args -> {
                    if (session.prediction == null) {
                        throw new IllegalArgumentException("Classify before retrieving evidence.");
                    }
                    var topK = args.path("top_k");
                    if (!topK.isIntegralNumber() || !topK.canConvertToInt()
                            || topK.intValue() < 3 || topK.intValue() > 15) {
                        throw new IllegalArgumentException("top_k must be an integer from 3 to 15.");
                    }
                    session.evidence = ml.retrieve(text, topK.intValue());
                    return session.evidence;
                }));
        tools.add(tool(session, "submit_route",
                "Finish by routing to the label supported by both classifier and retrieval.",
                """
                {"type":"object","properties":{"label":{"type":"string"},"reason":{"type":"string"}},
                 "required":["label","reason"],"additionalProperties":false}
                """,
                args -> {
                    String label = args.path("label").asText(null);
                    if (!session.supports(label)) {
                        throw new IllegalArgumentException("Evidence insufficient for this route. Request review.");
                    }
                    session.decision = new TriageSession.Decision("auto_route", label, reason(args));
                    return Map.of("status", "auto_route", "label", label);
                }));
        tools.add(tool(session, "request_review", "Finish and send this article to an editor. Give a brief reason.",
                """
                {"type":"object","properties":{"reason":{"type":"string"}},
                 "required":["reason"],"additionalProperties":false}
                """,
                args -> {
                    String label = session.prediction == null ? null : session.prediction.label();
                    session.decision = new TriageSession.Decision("human_review", label, reason(args));
                    return Map.of("status", "human_review");
                }));
        return tools;
    }

    private ToolCallback tool(TriageSession session, String name, String description,
                              String schema, Operation op) {
        return new ToolCallback() {
            @Override
            public ToolDefinition getToolDefinition() {
                return ToolDefinition.builder().name(name).description(description).inputSchema(schema).build();
            }

            @Override
            public String call(String input) {
                try {
                    requireOpen(session);
                    session.requireSkill();
                    JsonNode args = json.readTree(input);
                    if (args == null || !args.isObject()) {
                        return failure(session, name, "Tool arguments must be an object.");
                    }
                    Object observation = op.run(args);
                    session.trace.add(new TriageResult.ToolTrace(name, true, observation));
                    return encode(observation);
                } catch (IllegalArgumentException ex) {
                    return failure(session, name, ex.getMessage());
                } catch (IllegalStateException ex) {
                    return failure(session, name, ex.getMessage());
                } catch (JsonProcessingException ex) {
                    return failure(session, name, "Invalid tool JSON.");
                }
            }
        };
    }

    private static void requireOpen(TriageSession session) {
        if (session.decision != null) {
            throw new IllegalStateException("A final decision already exists. Respond without more tools.");
        }
    }

    static String reason(JsonNode args) {
        String value = args.path("reason").asText("").strip();
        int length = value.codePointCount(0, value.length());
        if (length < 5) {
            throw new IllegalArgumentException("Provide a reason of at least five characters.");
        }
        return length <= 600 ? value : value.substring(0, value.offsetByCodePoints(0, 599)) + "\u2026";
    }

    private String failure(TriageSession session, String name, String error) {
        var observation = Map.of("error", error);
        session.trace.add(new TriageResult.ToolTrace(name, false, observation));
        return encode(observation);
    }

    private String encode(Object value) {
        try {
            return json.writeValueAsString(value);
        } catch (JsonProcessingException ex) {
            throw new IllegalStateException("Tool result could not be serialized.", ex);
        }
    }

    @FunctionalInterface
    private interface Operation {
        Object run(JsonNode args);
    }
}
