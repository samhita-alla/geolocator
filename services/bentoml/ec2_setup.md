# EC2 Instance Setup

## Docker installation
- sudo yum update -y
- sudo amazon-linux-extras install docker -y
- sudo service docker start
- sudo usermod -a -G docker ec2-user
- Logout and login
- `docker info`

## Bento Server
- Go to the `Security` tab and add a HTTP inbound rule protocol
- `docker pull ghcr.io/samhita-alla/bento:0.0.1`
- `docker run -p 80:3000 ghcr.io/samhita-alla/bento:0.0.1`

## Test API
- `python services/bentoml/test_api.py --url <EC2_PUBLIC_IP>`
