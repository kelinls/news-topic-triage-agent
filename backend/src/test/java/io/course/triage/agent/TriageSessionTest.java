package io.course.triage.agent;

import org.junit.jupiter.api.Test;
import io.course.triage.ml.MlClient;
import com.agentx.ai.core.tools.SkillsTool;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.core.io.ClassPathResource;

import static org.junit.jupiter.api.Assertions.*;

class TriageSessionTest {
    @Test
    void boundsVerboseReasonsWithoutLosingTheDecision() {
        var reason = new ObjectMapper().createObjectNode().put("reason", "Policy evidence. ".repeat(80));
        assertEquals(600, TriageTools.reason(reason).length());
        assertThrows(IllegalArgumentException.class, () -> TriageTools.reason(reason.put("reason", " ")));
    }

    @Test
    void loadsSkillThroughFrameworkAndStripsLocalPath() {
        var skill = SkillsTool.builder()
                .addSkillsResource(new ClassPathResource("skills/text-classification/SKILL.md")).build();
        var session = new TriageSession(true);
        var callback = new TriageTools(new ObjectMapper(), null).bind(session, "", skill).getFirst();
        var result = callback.call("{\"command\":\"news-topic-classification\"}");
        assertTrue(result.contains("## Codebook"), result);
        assertFalse(result.contains("技能工作目录"));
        assertTrue(session.skillLoaded);
        assertTrue(session.trace.getFirst().ok());
    }

    @Test
    void autoRouteRequiresSkillAndTwoMatchingObservations() {
        var session = new TriageSession(true);
        var prediction = new MlClient.Prediction("comp.graphics", .91, .83, java.util.List.of());
        var matching = new MlClient.Evidence("comp.graphics", .70, java.util.Map.of());
        assertThrows(IllegalStateException.class, session::requireSkill);
        session.skillLoaded = true;
        session.requireSkill();
        assertFalse(session.supports("comp.graphics"));
        session.prediction = prediction;
        session.evidence = new MlClient.Evidence("sci.med", .70, java.util.Map.of());
        assertFalse(session.supports("comp.graphics"));
        session.evidence = matching;
        assertTrue(session.supports("comp.graphics"));
        assertFalse(session.supports("sci.med"));
    }
}
