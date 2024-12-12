from tqdm import tqdm


def train(model, train_loader, optimizer, loss_fn, device, num_epochs=20):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = 0

        for i, (noisy, clean) in enumerate(tqdm(train_loader, desc=f'Training epoch {epoch + 1}')):
            noisy = noisy.squeeze(1).to(device)
            clean = clean.squeeze(1).to(device)

            optimizer.zero_grad()

            masks = model(noisy)

            loss = sum([loss_fn(mask, clean) for mask in masks])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        epoch_loss = running_loss / total_batches
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')
