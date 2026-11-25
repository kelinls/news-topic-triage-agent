package io.course.triage.agent;

import java.util.ArrayList;
import java.util.List;

import io.course.triage.ml.MlClient;

final class TriageSession {
    final boolean useSkill;
    final List<TriageResult.ToolTrace> trace = new ArrayList<>();
    boolean skillLoaded;
    MlClient.Prediction prediction;
    MlClient.Evidence evidence;
    Decision decision;

    TriageSession(boolean useSkill) {
        this.useSkill = useSkill;
    }

    void requireSkill() {
        if (useSkill && !skillLoaded) {
            throw new IllegalStateException("Load news-topic-classification through Skill first.");
        }
    }

    boolean supports(String label) {
        return label != null && prediction != null && evidence != null
                && label.equals(prediction.label()) && label.equals(evidence.label())
                && prediction.confidence() >= .35 && prediction.margin() >= .08 && evidence.agreement() >= .55;
    }

    record Decision(String status, String label, String reason) {}
}
