---
name: skill-create
description: When users want to create a new OpenCode skill for a specific task
---
# Create Skill

Use this skill when a user wants to create a new OpenCode skill for a specific task or use case.

## Purpose

This is a meta-skill that helps create new skills for the OpenCode skills directory. It follows an iterative workflow to ensure high-quality skills that are well-tested and documented.

## Directory Structure

New skills should be created in: `.opencode/skills/{skill-name-here}/`

Each skill should contain a `SKILL.md` file with:
- YAML frontmatter at the top with `name` and `description`
- Clear instructions on when to use this skill
- Process steps for the skill's workflow
- Best practices and guidance

## Process

### 1. Capture Intent

Understand what new skill the user wants to create:
- What task or use case should the skill handle?
- What specific problem does it solve?
- When should it be triggered (based on description)?

### 2. Draft the Skill

Create the initial `SKILL.md` file with:
- **YAML frontmatter**: `name` (kebab-case) and `description`
- **Instructions**: Clear statement of when to use this skill
- **Process steps**: Numbered steps for the skill's workflow
- **Best practices**: Any important guidelines or tips

### 3. Create Test Cases

Develop test cases to validate the skill's effectiveness:
- **Positive cases**: Scenarios where the skill should be triggered
- **Negative cases**: Scenarios where the skill should NOT be triggered
- **Edge cases**: Boundary conditions and unusual inputs

### 4. Run Evaluations

Test the skill with and without it:
- **With skill**: Run the skill to see how it handles test cases
- **Without skill**: Run the same test cases without the skill
- **Compare results**: Document differences in behavior and outcomes

### 5. Generate Benchmarks

Use the eval-viewer to generate quantitative benchmarks:
- Run `eval-viewer/generate_review.py` to create a review
- Analyze success rates and response quality
- Document metrics for comparison

### 6. Iterate and Improve

Based on evaluation results:
- **Fix issues**: Address problems identified in testing
- **Refine instructions**: Improve clarity and precision
- **Update descriptions**: Optimize for better triggering accuracy
- **Add edge cases**: Handle previously unaddressed scenarios

## Best Practices

- **Clarity**: Make instructions unambiguous and easy to follow
- **Specificity**: Be specific about when the skill applies
- **Completeness**: Cover edge cases and error scenarios
- **Testability**: Ensure the skill can be evaluated objectively
- **Conciseness**: Keep instructions focused and avoid unnecessary complexity

## Example Workflow

1. User: "I want a skill that helps me write unit tests for React components"
2. Create: `.opencode/skills/react-unit-test/SKILL.md`
3. Draft: Initial version with basic workflow
4. Test: Create test cases for various React component types
5. Evaluate: Compare with and without the skill
6. Iterate: Refine based on results

## When NOT to Use This Skill

- If the task doesn't require a new skill (use existing skills)
- If the skill already exists
- If the request is too vague to define a useful skill
