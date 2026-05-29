"""In-process arka launcher with the per-row transform-stage patch applied.

Replaces `arka --config ...` when --skip-errors is requested. Imports arka,
applies the transform-stage patch directly (no import hooks, no recursion),
then calls arka.main(argv).
"""
from __future__ import annotations

import sys


def _apply_transform_skip_patch() -> None:
    from arka.pipeline import generator_stages as gen_mod
    ConversationRecord = gen_mod.ConversationRecord

    cls = gen_mod.TransformGeneratorStage
    flag = "_humanize_skip_errors_patched"
    if getattr(cls, flag, False):
        return

    def patched_run(self, records, ctx):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Mirror the upstream config-resolution chain.
        if self.config is None:
            if ctx.config and hasattr(ctx.config, "generator"):
                self.config = ctx.config.generator
            if self.config is None and ctx.config and hasattr(ctx.config, "pipeline"):
                for stage_cfg in ctx.config.pipeline:
                    if isinstance(stage_cfg, gen_mod.TransformGeneratorConfig):
                        self.config = stage_cfg
                        break
        if self.config is None:
            self.config = gen_mod.TransformGeneratorConfig(
                input_field="payload.instruction",
                output_field="payload.response",
            )
        transformable = [r for r in records if isinstance(r, ConversationRecord)]
        if not transformable:
            self._write_artifacts(ctx=ctx, dropped_records=[], costs=[])
            return []

        llm_client = self._llm_client or ctx.llm_client(
            override=self.config.llm_override
        )

        max_workers = max(1, getattr(ctx, "max_workers", 1) or 1)

        def _one(record):
            try:
                input_text = self._field_value(record, self.config.input_field)
                output = llm_client.complete_structured(
                    messages=self._messages_for_input(input_text, self.config),
                    schema=gen_mod.TransformResponse,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                parsed = output.parsed
                if not isinstance(parsed, gen_mod.TransformResponse):
                    raise ValueError(
                        "Transform output did not parse into TransformResponse"
                    )
                cost = output.usage.cost_usd
                new_record = self._build_transformed_record(
                    record=record,
                    transformed_text=parsed.text.strip(),
                    config_hash=self._config_hash(ctx),
                    generator_config=self.config,
                )
                return ("ok", record, new_record, cost)
            except Exception as exc:  # noqa: BLE001
                return ("err", record, exc, None)

        transformed: list = []
        dropped: list = []
        costs: list[float] = []
        n_err = 0
        total = len(transformable)
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_one, r) for r in transformable]
            for fut in as_completed(futures):
                status, rec, payload, cost = fut.result()
                done += 1
                if status == "ok":
                    transformed.append(payload)
                    if cost is not None:
                        costs.append(cost)
                else:
                    n_err += 1
                    dropped.append(rec)
                    if n_err <= 5:
                        print(
                            f"  ! transform skip-error: "
                            f"{type(payload).__name__}: {str(payload)[:160]}",
                            file=sys.stderr, flush=True,
                        )
                if done % 25 == 0 or done == total:
                    print(
                        f"  transform progress: {done}/{total} ok={len(transformed)} "
                        f"err={n_err}",
                        file=sys.stderr, flush=True,
                    )
        if n_err:
            print(
                f"  ! transform stage skipped {n_err}/{total} failed rows "
                f"(via _arka_runner.py)",
                file=sys.stderr, flush=True,
            )
        self._write_artifacts(ctx=ctx, dropped_records=dropped, costs=costs)
        return transformed

    cls.run = patched_run  # type: ignore[assignment]
    setattr(cls, flag, True)


def main() -> None:
    _apply_transform_skip_patch()
    from arka.cli import main as arka_main
    # arka.cli.main reads sys.argv; strip our prog name.
    arka_main(sys.argv[1:])


if __name__ == "__main__":
    main()
