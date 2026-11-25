package io.course.triage.api;

import java.util.Map;

import io.course.triage.agent.TriageResult;
import io.course.triage.agent.TriageService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
class TriageController {
    private final TriageService service;

    TriageController(TriageService service) {
        this.service = service;
    }

    @GetMapping("/health")
    Map<String, String> health() {
        return Map.of("status", "ok", "service", "news-triage");
    }

    @PostMapping("/triage")
    TriageResult triage(@RequestBody Request request) {
        return service.triage(request.text(), !Boolean.FALSE.equals(request.useSkill()));
    }

    record Request(String text, Boolean useSkill) {}
}
