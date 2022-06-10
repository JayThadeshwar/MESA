# Generated by Django 4.0.3 on 2022-06-10 09:01

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Chapter',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=500)),
                ('content', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('userId', models.AutoField(primary_key=True, serialize=False)),
                ('userName', models.CharField(max_length=500)),
                ('contactNumber', models.CharField(max_length=500)),
                ('emailId', models.CharField(max_length=500)),
                ('password', models.CharField(max_length=500)),
                ('dateOfBirth', models.DateField()),
                ('markForDelete', models.BooleanField(default=False)),
            ],
        ),
    ]
