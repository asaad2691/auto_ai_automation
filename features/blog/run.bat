@echo off
composer create-project --prefer-dist laravel/laravel blog_app
cd blog_app
php artisan make:controller BlogController
php artisan make:model Blog -m
mkdir resources\views\blog
type nul > resources\views\blog\index.blade.php