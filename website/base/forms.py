from django import forms


class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        label="",
        widget=forms.FileInput(
            attrs={
                "accept": "image/*",
                "class": "file-input-hidden",
            }
        ),
    )
