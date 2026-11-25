import torch


def get_bpr_loss(user_embedding, positive_embedding, negative_embedding):

    pos_score = torch.sum(torch.mul(user_embedding, positive_embedding), dim=1)

    neg_score = torch.sum(torch.mul(user_embedding, negative_embedding), dim=1)

    # loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))

    loss = - torch.log(torch.sigmoid(pos_score - neg_score) + 10e-8)

    return torch.mean(loss)


def get_reg_loss(*embeddings):
    reg_loss = 0
    for embedding in embeddings:
        reg_loss += 1 / 2 * embedding.norm(2).pow(2) / float(embedding.shape[0])

    return reg_loss


def get_InfoNCE_loss(embedding_1, embedding_2, temperature):
    embedding_1 = torch.nn.functional.normalize(embedding_1)
    embedding_2 = torch.nn.functional.normalize(embedding_2)

    pos_score = (embedding_1 * embedding_2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)

    ttl_score = torch.matmul(embedding_1, embedding_2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)

    cl_loss = - torch.log(pos_score / ttl_score + 10e-6)
    return torch.mean(cl_loss)

def get_sccf_loss(
    all_user_embeddings,
    all_item_embeddings,
    user_indices,
    positive_item_indices,
    temperature,
):

    if temperature <= 0:
        raise ValueError(
            "temperature must be positive, "
            f"but got {temperature}."
        )

    if user_indices.dim() != 1:
        raise ValueError(
            "user_indices must be one-dimensional."
        )

    if positive_item_indices.dim() != 1:
        raise ValueError(
            "positive_item_indices must be one-dimensional."
        )

    if user_indices.shape != positive_item_indices.shape:
        raise ValueError(
            "user_indices and positive_item_indices "
            "must have the same shape."
        )

    if user_indices.numel() <= 1:
        raise ValueError(
            "SCCF requires a batch size greater than one."
        )

    batch_user_embeddings = (
        all_user_embeddings[user_indices.long()]
    )
    batch_item_embeddings = (
        all_item_embeddings[positive_item_indices.long()]
    )

    batch_user_embeddings = (
        torch.nn.functional.normalize(
            batch_user_embeddings,
            dim=-1,
        )
    )
    batch_item_embeddings = (
        torch.nn.functional.normalize(
            batch_item_embeddings,
            dim=-1,
        )
    )

    positive_similarity = (
        batch_user_embeddings
        * batch_item_embeddings
    ).sum(dim=-1)

    positive_kernel = (
        torch.exp(
            positive_similarity / temperature
        )
        + torch.exp(
            positive_similarity.pow(2) / temperature
        )
    )

    positive_term = (
        torch.log(positive_kernel).mean()
    )

    unique_users, user_counts = torch.unique(
        user_indices.long(),
        return_counts=True,
    )
    unique_items, item_counts = torch.unique(
        positive_item_indices.long(),
        return_counts=True,
    )

    unique_user_embeddings = (
        all_user_embeddings[unique_users]
    )
    unique_item_embeddings = (
        all_item_embeddings[unique_items]
    )

    unique_user_embeddings = (
        torch.nn.functional.normalize(
            unique_user_embeddings,
            dim=-1,
        )
    )
    unique_item_embeddings = (
        torch.nn.functional.normalize(
            unique_item_embeddings,
            dim=-1,
        )
    )

    similarity_matrix = torch.matmul(
        unique_user_embeddings,
        unique_item_embeddings.transpose(0, 1),
    )

    all_pair_kernel = (
        torch.exp(
            similarity_matrix / temperature
        )
        + torch.exp(
            similarity_matrix.pow(2) / temperature
        )
    )

    pair_counts = torch.matmul(
        user_counts.reshape(-1, 1).to(
            dtype=all_pair_kernel.dtype
        ),
        item_counts.reshape(1, -1).to(
            dtype=all_pair_kernel.dtype
        ),
    )

    all_pair_term = torch.log(
        (
            all_pair_kernel
            * pair_counts
        ).sum()
        / pair_counts.sum()
    )

    sccf_loss = (
        -positive_term
        + all_pair_term
    )

    if not torch.isfinite(sccf_loss).item():
        raise RuntimeError(
            "SCCF loss is not finite."
        )

    return sccf_loss

def get_cross_view_sccf_loss(
    first_view_embeddings,
    second_view_embeddings,
    node_indices,
    temperature,
):

    if first_view_embeddings.dim() != 2:
        raise ValueError(
            "first_view_embeddings must be two-dimensional."
        )

    if second_view_embeddings.dim() != 2:
        raise ValueError(
            "second_view_embeddings must be two-dimensional."
        )

    if first_view_embeddings.shape != second_view_embeddings.shape:
        raise ValueError(
            "first_view_embeddings and second_view_embeddings "
            "must have the same shape."
        )

    if node_indices.dim() != 1:
        raise ValueError(
            "node_indices must be one-dimensional."
        )

    return get_sccf_loss(
        all_user_embeddings=first_view_embeddings,
        all_item_embeddings=second_view_embeddings,
        user_indices=node_indices.long(),
        positive_item_indices=node_indices.long(),
        temperature=temperature,
    )


def get_ELBO_loss(recon_x, x, mu, logvar, anneal):
    BCE = - torch.mean(torch.sum(torch.nn.functional.log_softmax(recon_x, 1) * x, -1))

    KLD = - 0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return BCE + anneal * KLD


def get_align_loss(embedding_1, embedding_2):
    embedding_1 = torch.nn.functional.normalize(embedding_1, dim=-1)
    embedding_2 = torch.nn.functional.normalize(embedding_2, dim=-1)
    return torch.mean((embedding_1 - embedding_2).norm(p=2, dim=1).pow(2))
   
   
def get_transR_loss(embedding_h, embedding_t, embedding_r):
    embedding_h = torch.nn.functional.normalize(embedding_h, dim=-1)
    embedding_t = torch.nn.functional.normalize(embedding_t, dim=-1)
    embedding_r = torch.nn.functional.normalize(embedding_r, dim=-1)
    return torch.mean((embedding_h + embedding_r - embedding_t).norm(p=2, dim=1).pow(2))


def get_uniform_loss(embedding):
    embedding = torch.nn.functional.normalize(embedding, dim=-1)
    return torch.pdist(embedding, p=2).pow(2).mul(-2).exp().mean().log()

