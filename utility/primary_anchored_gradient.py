import torch


def compose_primary_and_auxiliary_gradients(
    primary_gradient,
    auxiliary_gradient,
):

    if primary_gradient.shape != auxiliary_gradient.shape:
        raise RuntimeError(
            "Primary and auxiliary gradients must "
            "have identical shapes, but got "
            f"{tuple(primary_gradient.shape)} and "
            f"{tuple(auxiliary_gradient.shape)}."
        )

    if not torch.isfinite(
        primary_gradient
    ).all().item():
        raise FloatingPointError(
            "Primary gradient contains a non-finite value."
        )

    if not torch.isfinite(
        auxiliary_gradient
    ).all().item():
        raise FloatingPointError(
            "Auxiliary gradient contains a non-finite value."
        )

    processed_auxiliary_gradient = (
        auxiliary_gradient.detach().clone()
    )

    primary_squared_norm = torch.sum(
        primary_gradient
        * primary_gradient
    )
    anchor_available = (
        primary_squared_norm.item() > 0.0
    )
    projection_applied = False

    if anchor_available:
        primary_auxiliary_dot = torch.sum(
            primary_gradient
            * processed_auxiliary_gradient
        )

        if primary_auxiliary_dot.item() < 0.0:
            processed_auxiliary_gradient = (
                processed_auxiliary_gradient
                - (
                    primary_auxiliary_dot
                    / primary_squared_norm
                )
                * primary_gradient
            )
            projection_applied = True

    combined_gradient = (
        primary_gradient
        + processed_auxiliary_gradient
    )

    if not torch.isfinite(
        combined_gradient
    ).all().item():
        raise FloatingPointError(
            "Combined gradient contains a non-finite value."
        )

    return {
        "combined_gradient": (
            combined_gradient.detach().clone()
        ),
        "processed_auxiliary_gradient": (
            processed_auxiliary_gradient
            .detach()
            .clone()
        ),
        "anchor_available": anchor_available,
        "projection_applied": (
            projection_applied
        ),
    }

def _clone_current_gradients(
    parameters,
):
    return [
        (
            None
            if parameter.grad is None
            else parameter.grad.detach().clone()
        )
        for parameter in parameters
    ]


def apply_primary_anchored_backward(
    model,
    primary_loss,
    auxiliary_loss,
):

    model.zero_grad(set_to_none=True)

    if primary_loss.numel() != 1:
        raise RuntimeError(
            "Primary loss must be scalar."
        )

    if auxiliary_loss.numel() != 1:
        raise RuntimeError(
            "Auxiliary loss must be scalar."
        )

    if not primary_loss.requires_grad:
        raise RuntimeError(
            "Primary loss does not require gradients."
        )

    if not auxiliary_loss.requires_grad:
        raise RuntimeError(
            "Auxiliary loss does not require gradients."
        )

    if not torch.isfinite(
        primary_loss
    ).all().item():
        raise FloatingPointError(
            "Primary loss is non-finite."
        )

    if not torch.isfinite(
        auxiliary_loss
    ).all().item():
        raise FloatingPointError(
            "Auxiliary loss is non-finite."
        )

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if len(parameters) == 0:
        raise RuntimeError(
            "Model contains no trainable parameters."
        )

    statistics = {
        "trainable_parameter_count": (
            len(parameters)
        ),
        "both_used_count": 0,
        "primary_only_count": 0,
        "auxiliary_only_count": 0,
        "unused_count": 0,
        "zero_anchor_count": 0,
        "projection_count": 0,
        "assigned_gradient_count": 0,
    }

    try:
        model.zero_grad(set_to_none=True)
        primary_loss.backward(
            retain_graph=True
        )
        primary_gradients = (
            _clone_current_gradients(
                parameters
            )
        )

        model.zero_grad(set_to_none=True)
        auxiliary_loss.backward()
        auxiliary_gradients = (
            _clone_current_gradients(
                parameters
            )
        )

        model.zero_grad(set_to_none=True)

        for (
            parameter,
            primary_gradient,
            auxiliary_gradient,
        ) in zip(
            parameters,
            primary_gradients,
            auxiliary_gradients,
        ):
            if (
                primary_gradient is None
                and auxiliary_gradient is None
            ):
                parameter.grad = None
                statistics["unused_count"] += 1
                continue

            if primary_gradient is None:
                if not torch.isfinite(
                    auxiliary_gradient
                ).all().item():
                    raise FloatingPointError(
                        "An auxiliary-only gradient "
                        "contains a non-finite value."
                    )

                parameter.grad = (
                    auxiliary_gradient
                    .detach()
                    .clone()
                )
                statistics[
                    "auxiliary_only_count"
                ] += 1
                statistics[
                    "assigned_gradient_count"
                ] += 1
                continue

            if auxiliary_gradient is None:
                if not torch.isfinite(
                    primary_gradient
                ).all().item():
                    raise FloatingPointError(
                        "A Primary-only gradient "
                        "contains a non-finite value."
                    )

                parameter.grad = (
                    primary_gradient
                    .detach()
                    .clone()
                )
                statistics[
                    "primary_only_count"
                ] += 1
                statistics[
                    "assigned_gradient_count"
                ] += 1
                continue

            statistics["both_used_count"] += 1

            composition = (
                compose_primary_and_auxiliary_gradients(
                    primary_gradient=(
                        primary_gradient
                    ),
                    auxiliary_gradient=(
                        auxiliary_gradient
                    ),
                )
            )

            parameter.grad = (
                composition[
                    "combined_gradient"
                ].detach().clone()
            )
            statistics[
                "assigned_gradient_count"
            ] += 1

            if not composition[
                "anchor_available"
            ]:
                statistics[
                    "zero_anchor_count"
                ] += 1

            if composition[
                "projection_applied"
            ]:
                statistics[
                    "projection_count"
                ] += 1

        if statistics[
            "assigned_gradient_count"
        ] == 0:
            raise RuntimeError(
                "No gradients were assigned."
            )

        return statistics

    except Exception:
        model.zero_grad(set_to_none=True)
        raise
