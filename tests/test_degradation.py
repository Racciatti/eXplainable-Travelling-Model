import numpy as np
import pytest

from classes.degradation import apply_gaussian_blur, apply_label_noise


def test_label_noise_changes_every_label_when_p_is_one():
    labels = np.array([0, 1, 2, 0, 1, 2] * 10)
    noisy = apply_label_noise(labels, 1.0)
    assert noisy.shape == labels.shape
    assert np.all(noisy != labels)
    assert set(noisy) <= {0, 1, 2}


def test_label_noise_does_not_mutate_input_and_keeps_shape():
    labels = np.array([[0], [1], [0], [1]])
    original = labels.copy()
    noisy = apply_label_noise(labels, 0.5)
    assert np.array_equal(labels, original)
    assert noisy.shape == (4, 1)


def test_label_noise_single_class_is_noop():
    labels = np.zeros(5, dtype=int)
    assert np.array_equal(apply_label_noise(labels, 0.9), labels)


@pytest.mark.parametrize("p", [0.0, -0.1, 1.5])
def test_label_noise_rejects_invalid_probability(p):
    with pytest.raises(ValueError):
        apply_label_noise(np.array([0, 1]), p)


def test_label_noise_rate_is_close_to_p():
    np.random.seed(0)
    labels = np.random.randint(0, 2, size=5000)
    noisy = apply_label_noise(labels, 0.3)
    assert abs(np.mean(noisy != labels) - 0.3) < 0.03


def test_blur_preserves_constant_image_and_dtype():
    images = np.full((2, 6, 6), 100, dtype=np.uint8)
    out = apply_gaussian_blur(images, kernel_size=3)
    assert out.dtype == images.dtype
    assert np.all(out == 100)


def test_blur_smooths_an_impulse():
    img = np.zeros((1, 7, 7), dtype=float)
    img[0, 3, 3] = 1.0
    out = apply_gaussian_blur(img, kernel_size=3, sigma=1.0)
    assert out[0, 3, 3] < 1.0
    assert out[0, 3, 4] > 0.0
    assert out.sum() == pytest.approx(1.0)


def test_blur_supports_color_images():
    images = np.random.randint(0, 255, size=(1, 5, 5, 3), dtype=np.uint8)
    assert apply_gaussian_blur(images, kernel_size=3).shape == images.shape


@pytest.mark.parametrize("k", [1, 2, 4])
def test_blur_rejects_bad_kernel(k):
    with pytest.raises(ValueError):
        apply_gaussian_blur(np.zeros((1, 4, 4)), kernel_size=k)


def test_blur_rejects_bad_rank():
    with pytest.raises(ValueError):
        apply_gaussian_blur(np.zeros((4, 4)), kernel_size=3)
