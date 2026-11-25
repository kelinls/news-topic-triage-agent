package io.course.triage.agent;

import java.util.List;

public record TriageResult(
        String status, String label, String reason, String model, String runtime,
        boolean skillLoaded, boolean fallback, long elapsedMs, List<ToolTrace> toolTrace) {
    public record ToolTrace(String tool, boolean ok, Object observation) {}
}
