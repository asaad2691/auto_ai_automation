  {
      "require": {
          "laravel/framework": "^8.0"
      },
      "autoload": {
          "psr-4": {
              "App\\": "app/"
          }
      },
      "scripts": {
          "post-root-package-install": [
              "@php -r \"file_exists('.env') || copy('.env.example', '.env');\""
          ],
          "post-create-project-cmd": [
              "@php artisan key:generate"
          ]
      }
  }
