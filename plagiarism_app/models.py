from django.db import models

class PlagiarismCheck(models.Model):
    """Model to store plagiarism check data"""
    text = models.TextField(verbose_name="Original Text")
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Plagiarism Check"
        verbose_name_plural = "Plagiarism Checks"
    
    def __str__(self):
        # Truncate text for display purposes
        max_length = 50
        if len(self.text) > max_length:
            return f"{self.text[:max_length]}..."
        return self.text
