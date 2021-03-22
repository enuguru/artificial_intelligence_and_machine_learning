/* global wpforms_settings,grecaptcha,wpforms_validate,wpforms_timepicker */

;(function($) {

	'use strict';

	var WPForms = {

		/**
		 * Start the engine.
		 *
		 * @since 1.2.3
		 */
		init: function() {

			// Document ready
			$(document).ready(WPForms.ready);

			// Page load
			$(window).on('load', WPForms.load);

			WPForms.bindUIActions();
			WPForms.bindOptinMonster();
		},

		/**
		 * Document ready.
		 *
		 * @since 1.2.3
		 */
		ready: function() {

			// Set user identifier
			WPForms.setUserIndentifier();

			WPForms.loadValidation();
			WPForms.loadDatePicker();
			WPForms.loadTimePicker();
			WPForms.loadInputMask();
			WPForms.loadPayments();

			// Randomize elements.
			$( '.wpforms-randomize' ).each( function() {
				var $list      = $( this ),
					$listItems = $list.children();
				while ( $listItems.length ) {
					$list.append( $listItems.splice( Math.floor( Math.random() * $listItems.length ), 1 )[0] );
				}
			});

			$(document).trigger('wpformsReady');
		},

		/**
		 * Page load.
		 *
		 * @since 1.2.3
		 */
		load: function() {

		},

		//--------------------------------------------------------------------//
		// Initializing
		//--------------------------------------------------------------------//

		/**
		 * Load jQuery Validation.
		 *
		 * @since 1.2.3
		 */
		loadValidation: function() {

			// Only load if jQuery validation library exists
			if (typeof $.fn.validate !== 'undefined') {

				// jQuery Validation library will not correctly validate
				// fields that do not have a name attribute, so we use the
				// `wpforms-input-temp-name` class to add a temporary name
				// attribute before validation is initialized, then remove it
				// before the form submits.
				$( '.wpforms-input-temp-name' ).each(function( index, el ) {
					var random = Math.floor( Math.random() * 9999 ) + 1;
					$( this ).attr( 'name', 'wpf-temp-' + random );
				});

				// Prepend URL field contents with http:// if user input doesn't contain a schema.
				$( '.wpforms-validate input[type=url]' ).change( function () {
					var url = $( this ).val();
					if ( ! url ) {
						return false;
					}
					if ( url.substr( 0, 7 ) !== 'http://' && url.substr( 0, 8 ) !== 'https://' ) {
						$( this ).val( 'http://' + url );
					}
				});

				$.validator.messages.required = wpforms_settings.val_required;
				$.validator.messages.url = wpforms_settings.val_url;
				$.validator.messages.email = wpforms_settings.val_email;
				$.validator.messages.number = wpforms_settings.val_number;

				// Payments: Validate method for Credit Card Number
				if(typeof $.fn.payment !== 'undefined') {
					$.validator.addMethod( "creditcard", function(value, element) {
						//var type  = $.payment.cardType(value);
						var valid = $.payment.validateCardNumber(value);
						return this.optional(element) || valid;
					}, wpforms_settings.val_creditcard);
					// @todo validate CVC and expiration
				}

				// Validate method for file extensions
				$.validator.addMethod( "extension", function(value, element, param) {
					param = typeof param === "string" ? param.replace( /,/g, "|" ) : "png|jpe?g|gif";
					return this.optional(element) || value.match( new RegExp( "\\.(" + param + ")$", "i" ) );
				}, wpforms_settings.val_fileextension );

				// Validate method for file size
				$.validator.addMethod("maxsize", function(value, element, param) {
					var maxSize = param,
						optionalValue = this.optional(element),
						i, len, file;
					if (optionalValue) {
						return optionalValue;
					}
					if (element.files && element.files.length) {
						i = 0;
						len = element.files.length;
						for (; i < len; i++) {
							file = element.files[i];
							if (file.size > maxSize) {
								return false;
							}
						}
					}
					return true;
				}, wpforms_settings.val_filesize);

				// Validate email addresses
				$.validator.methods.email = function( value, element ) {
					return this.optional( element ) || /^[a-z0-9.!#$%&'*+\/=?^_`{|}~-]+@((?=[a-z0-9-]{1,63}\.)(xn--)?[a-z0-9]+(-[a-z0-9]+)*\.)+[a-z]{2,63}$/i.test( value );
				};

				// Validate confirmations
				$.validator.addMethod("confirm", function(value, element, param) {
					return $.validator.methods.equalTo.call(this, value, element, param);
				}, wpforms_settings.val_confirm);

				// Validate required payments
				$.validator.addMethod( "required-payment", function( value, element ) {
					return WPForms.amountSanitize( value ) > 0;
				}, wpforms_settings.val_requiredpayment );

				// Validate 12-hour time
				$.validator.addMethod( "time12h", function( value, element ) {
					return this.optional( element ) || /^((0?[1-9]|1[012])(:[0-5]\d){1,2}(\ ?[AP]M))$/i.test( value );
				}, wpforms_settings.val_time12h );

				// Validate 24-hour time
				$.validator.addMethod( "time24h", function( value, element ) {
					return this.optional(element) || /^(([0-1]?[0-9])|([2][0-3])):([0-5]?[0-9])(\ ?[AP]M)?$/i.test(value);
				}, wpforms_settings.val_time24h );

				// Finally load jQuery Validation library for our forms
				$('.wpforms-validate').each(function() {
					var form   = $(this),
						formID = form.data('formid'),
						properties;

					// TODO: cleanup this BC with wpforms_validate.
					if (typeof window['wpforms_'+formID] !== 'undefined' && window['wpforms_'+formID].hasOwnProperty('validate')) {
						properties = window['wpforms_'+formID].validate;
					} else if ( typeof wpforms_validate !== 'undefined') {
						properties = wpforms_validate;
					} else {
						properties = {
							errorClass: 'wpforms-error',
							validClass: 'wpforms-valid',
							errorPlacement: function( error, element ) {
								if ( element.attr( 'type' ) === 'radio' || element.attr( 'type' ) === 'checkbox' ) {
									if ( element.hasClass( 'wpforms-likert-scale-option' ) ) {
										if ( element.closest( 'table' ).hasClass( 'single-row' ) ) {
											element.closest( 'table' ).after( error );
										} else {
											element.closest( 'tr' ).find( 'th' ).append( error );
										}
									} else if ( element.hasClass( 'wpforms-net-promoter-score-option' ) ) {
										element.closest( 'table' ).after( error );
									} else {
										element.parent().parent().parent().append( error );
									}
								} else if ( element.is( 'select' ) && element.attr( 'class' ).match( /date-month|date-day|date-year/ ) ) {
									if ( element.parent().find( 'label.wpforms-error:visible' ).length === 0 ) {
										element.parent().find( 'select:last' ).after( error );
									}
								} else {
									error.insertAfter(element);
								}
							},
							highlight: function( element, errorClass, validClass ) {
								var $element  = $( element ),
									$field    = $element.closest( '.wpforms-field' ),
									inputName = $element.attr( 'name' );
								if ( $element.attr( 'type' ) === 'radio' || $element.attr( 'type' ) === 'checkbox' ) {
									$field.find( 'input[name=\''+inputName+'\']' ).addClass( errorClass ).removeClass( validClass );
								} else {
									$element.addClass( errorClass ).removeClass( validClass );
								}
								$field.addClass( 'wpforms-has-error' );
							},
							unhighlight: function( element, errorClass, validClass ) {
								var $element  = $( element ),
									$field    = $element.closest( '.wpforms-field' ),
									inputName = $element.attr( 'name' );
								if ( $element.attr( 'type' ) === 'radio' || $element.attr( 'type' ) === 'checkbox' ) {
									$field.find( 'input[name=\''+inputName+'\']' ).addClass( validClass ).removeClass( errorClass );
								} else {
									$element.addClass( validClass ).removeClass( errorClass );
								}
								$field.removeClass( 'wpforms-has-error' );
							},
							submitHandler: function(form) {

								var $form   = $(form),
									$submit = $form.find('.wpforms-submit'),
									altText = $submit.data('alt-text');

								if ( WPForms.empty( $submit.get(0).recaptchaID ) && $submit.get(0).recaptchaID !== 0 ) {
									// Normal form.
									if (altText) {
										$submit.text(altText).prop('disabled', true);
									}

									// Remove name attributes if needed.
									$( '.wpforms-input-temp-name' ).removeAttr('name');

									form.submit();

								} else {
									// Form contains invisible reCAPTCHA.
									grecaptcha.execute( $submit.get(0).recaptchaID );
								}
							},
							onkeyup: function( element, event ) {
								// This code is copied from JQuery Validate 'onkeyup' method with only one change: 'wpforms-novalidate-onkeyup' class check.
								var excludedKeys = [ 16, 17, 18, 20, 35, 36, 37, 38, 39, 40, 45, 144, 225 ];

								if ( $( element ).hasClass('wpforms-novalidate-onkeyup') ) {
									return; // Disable onkeyup validation for some elements (e.g. remote calls).
								}

								if ( event.which === 9 && this.elementValue( element ) === "" || $.inArray( event.keyCode, excludedKeys ) !== -1 ) {
									return;
								} else if ( element.name in this.submitted || element.name in this.invalid ) {
									this.element( element );
								}
							},
							onfocusout: function( element ) {
								// This code is copied from JQuery Validate 'onfocusout' method with only one change: 'wpforms-novalidate-onkeyup' class check.
								var validate = false;

								if ( $( element ).hasClass('wpforms-novalidate-onkeyup') && !element.value ) {
									validate = true; // Empty value error handling for elements with onkeyup validation disabled.
								}

								if ( !this.checkable( element ) && ( element.name in this.submitted || !this.optional( element ) ) ) {
									validate = true;
								}

								if ( validate ) {
									this.element( element );
								}
							},
						}
					}
					form.validate( properties );
				});
			}
		},

		/**
		 * Load jQuery Date Picker.
		 *
		 * @since 1.2.3
		 */
		loadDatePicker: function() {

			// Only load if jQuery datepicker library exists
			if (typeof $.fn.flatpickr !== 'undefined') {
				$('.wpforms-datepicker').each(function() {
					var element = $(this),
						form    = element.closest('.wpforms-form'),
						formID  = form.data('formid'),
						fieldID = element.closest('.wpforms-field').data('field-id'),
						properties;

					if (typeof window['wpforms_'+formID+'_'+fieldID] !== 'undefined' && window['wpforms_'+formID+'_'+fieldID].hasOwnProperty('datepicker')) {
						properties = window['wpforms_'+formID+'_'+fieldID].datepicker;
					} else if (typeof window['wpforms_'+formID] !== 'undefined' && window['wpforms_'+formID].hasOwnProperty('datepicker')) {
						properties = window['wpforms_'+formID].datepicker;
					} else if (typeof wpforms_datepicker !== 'undefined') {
						properties = wpforms_datepicker;
					} else {
						properties = {
							disableMobile: true
						};
					}

					// Redefine locale only if user doesn't do that manually and we have the locale.
					if (
						! properties.hasOwnProperty( 'locale' ) &&
						typeof wpforms_settings !== 'undefined' &&
						wpforms_settings.hasOwnProperty( 'locale' )
					) {
						properties.locale = wpforms_settings.locale;
					}

					element.flatpickr(properties)
				});
			}
		},

		/**
		 * Load jQuery Time Picker.
		 *
		 * @since 1.2.3
		 */
		loadTimePicker: function() {

			// Only load if jQuery timepicker library exists
			if (typeof $.fn.timepicker !== 'undefined') {
				$('.wpforms-timepicker').each(function() {
					var element = $(this),
						form    = element.closest('.wpforms-form'),
						formID  = form.data('formid'),
						fieldID = element.closest('.wpforms-field').data('field-id'),
						properties;

					if (
						typeof window['wpforms_'+formID+'_'+fieldID] !== 'undefined' &&
						window['wpforms_'+formID+'_'+fieldID].hasOwnProperty('timepicker')
					) {
						properties = window['wpforms_'+formID+'_'+fieldID].timepicker;
					} else if (
						typeof window['wpforms_'+formID] !== 'undefined' &&
						window['wpforms_'+formID].hasOwnProperty('timepicker')
					) {
						properties = window['wpforms_'+formID].timepicker;
					} else if ( typeof wpforms_timepicker !== 'undefined') {
						properties = wpforms_timepicker;
					} else {
						properties = {
							scrollDefault: 'now',
							forceRoundTime: true
						};
					}

					element.timepicker(properties);
				});
			}
		},

		/**
		 * Load jQuery input masks.
		 *
		 * @since 1.2.3
		 */
		loadInputMask: function() {

			// Only load if jQuery input mask library exists
			if (typeof $.fn.inputmask !== 'undefined') {
				$('.wpforms-masked-input').inputmask();
			}
		},

		/**
		 * Payments: Do various payment-related tasks on load.
		 *
		 * @since 1.2.6
		 */
		loadPayments: function() {

			// Update Total field(s) with latest calculation
			$('.wpforms-payment-total').each(function(index, el) {
				WPForms.amountTotal(this);
			});

			// Credit card validation
			if(typeof $.fn.payment !== 'undefined') {
				$('.wpforms-field-credit-card-cardnumber').payment('formatCardNumber');
				$('.wpforms-field-credit-card-cardcvc').payment('formatCardCVC');
			}
		},

		//--------------------------------------------------------------------//
		// Binds.
		//--------------------------------------------------------------------//

		/**
		 * Element bindings.
		 *
		 * @since 1.2.3
		 */
		bindUIActions: function() {

			// Pagebreak navigation.
			$(document).on('click', '.wpforms-page-button', function(event) {
				event.preventDefault();
				WPForms.pagebreakNav($(this));
			});

			// Payments: Update Total field(s) when latest calculation.
			$(document).on('change input', '.wpforms-payment-price', function() {
				WPForms.amountTotal(this, true);
			});

			// Payments: Restrict user input payment fields.
			$(document).on('input', '.wpforms-payment-user-input', function() {
				var $this = $(this),
					amount = $this.val();
				$this.val(amount.replace(/[^0-9.,]/g, ''));
			});

			// Payments: Sanitize/format user input amounts.
			$(document).on('focusout', '.wpforms-payment-user-input', function() {
				var $this     = $(this),
					amount    = $this.val(),
					sanitized = WPForms.amountSanitize(amount),
					formatted = WPForms.amountFormat(sanitized);
				$this.val(formatted);
			});

			// Payment radio/checkbox fields: preselect the selected payment (from dynamic/fallback population).
			$( document ).ready( function () {
				// Radios.
				$( '.wpforms-field-radio .wpforms-image-choices-item input:checked' ).change();
				$( '.wpforms-field-payment-multiple .wpforms-image-choices-item input:checked' ).change();
				// Checkboxes.
				$( '.wpforms-field-checkbox .wpforms-image-choices-item input' ).change();
				$( '.wpforms-field-payment-checkbox .wpforms-image-choices-item input' ).change();
			} );

			// Rating field: hover effect.
			$( '.wpforms-field-rating-item' ).hover(
				function() {
					$( this ).parent().find( '.wpforms-field-rating-item' ).removeClass( 'selected hover' );
					$( this ).prevAll().andSelf().addClass( 'hover' );
				},
				function() {
					$( this ).parent().find( '.wpforms-field-rating-item' ).removeClass( 'selected hover' );
					$( this ).parent().find( 'input:checked' ).parent().prevAll().andSelf().addClass( 'selected' );
				}
			);

			// Rating field: toggle selected state.
			$( document ).on( 'change', '.wpforms-field-rating-item input', function() {

				var $this  = $( this ),
					$wrap  = $this.closest( '.wpforms-field-rating-items' ),
					$items = $wrap.find( '.wpforms-field-rating-item' );

				$items.removeClass( 'hover selected' );
				$this.parent().prevAll().andSelf().addClass( 'selected' );
			} );

			// Rating field: preselect the selected rating (from dynamic/fallback population).
			$( document ).ready( function () {
				$( '.wpforms-field-rating-item input:checked' ).change();
			} );

			// Checkbox/Radio/Payment checkbox: toggle selected state class.
			$( document ).on( 'change', '.wpforms-field-checkbox input, .wpforms-field-radio input, .wpforms-field-payment-multiple input, .wpforms-field-payment-checkbox input, .wpforms-field-gdpr-checkbox input', function( event ) {

				var $this  = $( this ),
					$field = $this.closest( '.wpforms-field' );

				if ( $field.hasClass( 'wpforms-conditional-hide' ) ) {
					event.preventDefault();
					return false;
				}

				switch ( $this.attr( 'type' ) ) {
					case 'radio':
						$this.closest( 'ul' ).find( 'li' ).removeClass( 'wpforms-selected' ).find('input[type=radio]').removeProp( 'checked' );
						$this
							.prop( 'checked', true )
							.closest( 'li' ).addClass( 'wpforms-selected' );
						break;

					case 'checkbox':
						if ( $this.prop( 'checked' ) ) {
							$this.closest( 'li' ).addClass( 'wpforms-selected' );
							$this.prop( 'checked', true );
						} else {
							$this.closest( 'li' ).removeClass( 'wpforms-selected' );
							$this.removeProp( 'checked' );
						}
						break;
				}
			} );
		},

		/**
		 * Update Pagebreak navigation.
		 *
		 * @since 1.2.2
		 */
		pagebreakNav: function(el) {

			var $this      = $(el),
				valid      = true,
				action     = $this.data('action'),
				page       = $this.data('page'),
				page2      = page,
				next       = page+1,
				prev       = page-1,
				formID     = $this.data('formid'),
				$form      = $this.closest('.wpforms-form'),
				$page      = $form.find('.wpforms-page-'+page),
				$submit    = $form.find('.wpforms-submit-container'),
				$indicator = $form.find('.wpforms-page-indicator'),
				$reCAPTCHA = $form.find('.wpforms-recaptcha-container'),
				pageScroll = false;

			// Page scroll
			// TODO: cleanup this BC with wpform_pageScroll.
			if ( window.wpforms_pageScroll === false ) {
				pageScroll = false;
			} else if ( !WPForms.empty( window.wpform_pageScroll ) ) {
				pageScroll = window.wpform_pageScroll;
			} else {
				pageScroll = 75;
			}

			// Toggling between pages
			if ( action === 'next' ){
				// Validate
				if (typeof $.fn.validate !== 'undefined') {
					$page.find('input.wpforms-field-required, select.wpforms-field-required, textarea.wpforms-field-required, .wpforms-field-required input, .wpforms-field-file-upload input').each(function(index, el) {
						if ( ! $(el).valid() ) {
							valid = false;
						}
					});
					// Scroll to first/top error on page
					var $topError = $page.find('.wpforms-error').first();
					if ($topError.length) {
						$('html, body').animate({
							scrollTop: $topError.offset().top-75
						}, 750, function() {
							$topError.focus();
						});
					}
				}
				// Move to next page
				if (valid) {
					page2 = next;
					$page.hide();
					var $nextPage = $form.find('.wpforms-page-'+next);
					$nextPage.show();
					if ( $nextPage.hasClass('last') ) {
						$reCAPTCHA.show();
						$submit.show();
					}
					if ( pageScroll ) {
						// Scroll to top of the form
						$('html, body').animate({
							scrollTop: $form.offset().top-pageScroll
						}, 1000);
					}
					$this.trigger('wpformsPageChange', [ page2, $form ] );
				}
			} else if ( action === 'prev' ) {
				// Move to prev page
				page2 = prev;
				$page.hide();
				$form.find('.wpforms-page-'+prev).show();
				$reCAPTCHA.hide();
				$submit.hide();
				if ( pageScroll ) {
					// Scroll to top of the form
					$('html, body').animate({
						scrollTop: $form.offset().top-pageScroll
					}, 1000);
				}
				$this.trigger('wpformsPageChange', [ page2, $form ] );
			}

			if ( $indicator ) {
				var theme = $indicator.data('indicator'),
					color = $indicator.data('indicator-color');
				if ('connector' === theme || 'circles' === theme) {
					$indicator.find('.wpforms-page-indicator-page').removeClass('active');
					$indicator.find('.wpforms-page-indicator-page-'+page2).addClass('active');
					$indicator.find('.wpforms-page-indicator-page-number').removeAttr('style');
					$indicator.find('.active .wpforms-page-indicator-page-number').css('background-color', color);
					if ( 'connector' === theme) {
						$indicator.find('.wpforms-page-indicator-page-triangle').removeAttr('style');
						$indicator.find('.active .wpforms-page-indicator-page-triangle').css('border-top-color', color);
					}
				} else if ('progress' === theme) {
					var $pageTitle = $indicator.find('.wpforms-page-indicator-page-title'),
						$pageSep   = $indicator.find('.wpforms-page-indicator-page-title-sep'),
						totalPages = $form.find('.wpforms-page').length,
						width = (page2/totalPages)*100;
					$indicator.find('.wpforms-page-indicator-page-progress').css('width', width+'%');
					$indicator.find('.wpforms-page-indicator-steps-current').text(page2);
					if ($pageTitle.data('page-'+page2+'-title')) {
						$pageTitle.css('display','inline').text($pageTitle.data('page-'+page2+'-title'));
						$pageSep.css('display','inline');
					} else {
						$pageTitle.css('display','none');
						$pageSep.css('display','none');
					}
				}
			}
		},

		/**
		 * OptinMonster compatibility.
		 *
		 * Re-initialize after OptinMonster loads to accomodate changes that
		 * have occurred to the DOM.
		 *
		 * @since 1.5.0
		 */
		bindOptinMonster: function() {

			// OM v5.
			document.addEventListener( 'om.Campaign.load', function( event ) {
				WPForms.ready();
				WPForms.optinMonsterRecaptchaReset( event.detail.Campaign.data.id );
			} );

			// OM Legacy.
			$( document ).on( 'OptinMonsterOnShow', function( event, data, object ) {
				WPForms.ready();
				WPForms.optinMonsterRecaptchaReset( data.optin );
			} );
		},

		/**
		 * Reset/recreate reCAPTCHA v2 inside OptinMonster.
		 *
		 * @since 1.5.0
		 */
		optinMonsterRecaptchaReset: function ( optinId ) {

			var $form               = $( '#om-' + optinId ).find( '.wpforms-form' ),
				$recaptchaContainer = $form.find( '.wpforms-recaptcha-container' ),
				$recaptcha          = $form.find( '.g-recaptcha' ),
				recaptchaSiteKey    = $recaptcha.attr( 'data-sitekey' ),
				recaptchaID         = 'recaptcha-' + Date.now();

			if ( $form.length && $recaptcha.length ) {

				$recaptcha.remove();
				$recaptchaContainer.prepend( '<div class="g-recaptcha" id="' + recaptchaID + '" data-sitekey="' + recaptchaSiteKey + '"></div>' );

				grecaptcha.render(
					recaptchaID,
					{
						sitekey: recaptchaSiteKey,
						callback: function(){
							wpformsRecaptchaCallback( $( '#' + recaptchaID ) );
						}
					}
				);
			}
		},

		//--------------------------------------------------------------------//
		// Other functions.
		//--------------------------------------------------------------------//

		/**
		 * Payments: Calculate total.
		 *
		 * @since 1.2.3
		 * @since 1.5.1 Added support for payment-checkbox field.
		 */
		amountTotal: function ( el, validate ) {

			var validate = validate || false,
				$form = $( el ).closest( '.wpforms-form' ),
				total = 0,
				totalFormatted = 0,
				totalFormattedSymbol = 0,
				currency = WPForms.getCurrency();

			$( '.wpforms-payment-price', $form ).each( function ( index, el ) {

				var amount = 0,
					$this = $( this );

				if ( $this.attr( 'type' ) === 'text' || $this.attr( 'type' ) === 'hidden' ) {
					amount = $this.val();
				}
				else if ( ( $this.attr( 'type' ) === 'radio' || $this.attr( 'type' ) === 'checkbox' ) && $this.is( ':checked' ) ) {
					amount = $this.data( 'amount' );
				}
				else if ( $this.is( 'select' ) && $this.find( 'option:selected' ).length > 0 ) {
					amount = $this.find( 'option:selected' ).data( 'amount' );
				}
				if ( ! WPForms.empty( amount ) ) {
					amount = WPForms.amountSanitize( amount );
					total = Number( total ) + Number( amount );
				}
			} );

			totalFormatted = WPForms.amountFormat( total );

			if ( 'left' === currency.symbol_pos ) {
				totalFormattedSymbol = currency.symbol + ' ' + totalFormatted;
			}
			else {
				totalFormattedSymbol = totalFormatted + ' ' + currency.symbol;
			}

			$form.find( '.wpforms-payment-total' ).each( function ( index, el ) {
				if ( 'hidden' === $( this ).attr( 'type' ) || 'text' === $( this ).attr( 'type' ) ) {
					$( this ).val( totalFormattedSymbol );
					if ( 'text' === $( this ).attr( 'type' ) && validate && $form.data( 'validator' ) ) {
						$( this ).valid();
					}
				} else {
					$( this ).text( totalFormattedSymbol );
				}
			} );
		},

		/**
		 * Sanitize amount and convert to standard format for calculations.
		 *
		 * @since 1.2.6
		 */
		amountSanitize: function(amount) {

			var currency = WPForms.getCurrency();

			amount = amount.toString().replace(/[^0-9.,]/g,'');

			if ( currency.decimal_sep === ',' && ( amount.indexOf(currency.decimal_sep) !== -1 ) ) {
				if ( currency.thousands_sep === '.' && amount.indexOf(currency.thousands_sep) !== -1 ) {
					amount = amount.replace(currency.thousands_sep,'');
				} else if( currency.thousands_sep === '' && amount.indexOf('.') !== -1 ) {
					amount = amount.replace('.','');
				}
				amount = amount.replace(currency.decimal_sep,'.');
			} else if ( currency.thousands_sep === ',' && ( amount.indexOf(currency.thousands_sep) !== -1 ) ) {
				amount = amount.replace(currency.thousands_sep,'');
			}

			return WPForms.numberFormat( amount, 2, '.', '' );
		},

		/**
		 * Format amount.
		 *
		 * @since 1.2.6
		 */
		amountFormat: function(amount) {

			var currency = WPForms.getCurrency();

			amount = String(amount);

			// Format the amount
			if ( currency.decimal_sep === ',' && ( amount.indexOf(currency.decimal_sep) !== -1 ) ) {
				var sepFound = amount.indexOf(currency.decimal_sep),
					whole    = amount.substr(0, sepFound),
					part     = amount.substr(sepFound+1, amount.strlen-1);
				amount = whole + '.' + part;
			}

			// Strip , from the amount (if set as the thousands separator)
			if ( currency.thousands_sep === ',' && ( amount.indexOf(currency.thousands_sep) !== -1 ) ) {
				amount = amount.replace(',','');
			}

			if ( WPForms.empty( amount ) ) {
				amount = 0;
			}

			return WPForms.numberFormat( amount, 2, currency.decimal_sep, currency.thousands_sep );
		},

		/**
		 * Get site currency settings.
		 *
		 * @since 1.2.6
		 */
		getCurrency: function() {

			var currency = {
				code: 'USD',
				thousands_sep: ',',
				decimal_sep: '.',
				symbol: '$',
				symbol_pos: 'left'
			};

			// Backwards compatibility.
			if ( typeof wpforms_settings.currency_code !== 'undefined' ) {
				currency.code = wpforms_settings.currency_code;
			}
			if ( typeof wpforms_settings.currency_thousands !== 'undefined' ) {
				currency.thousands_sep = wpforms_settings.currency_thousands;
			}
			if ( typeof wpforms_settings.currency_decimal !== 'undefined' ) {
				currency.decimal_sep = wpforms_settings.currency_decimal;
			}
			if ( typeof wpforms_settings.currency_symbol !== 'undefined' ) {
				currency.symbol = wpforms_settings.currency_symbol;
			}
			if ( typeof wpforms_settings.currency_symbol_pos !== 'undefined' ) {
				currency.symbol_pos = wpforms_settings.currency_symbol_pos;
			}

			return currency;
		},

		/**
		 * Format number.
		 *
		 * @link http://locutus.io/php/number_format/
		 * @since 1.2.6
		 */
		numberFormat: function (number, decimals, decimalSep, thousandsSep) {

			number = (number + '').replace(/[^0-9+\-Ee.]/g, '');
			var n = !isFinite(+number) ? 0 : +number;
			var prec = !isFinite(+decimals) ? 0 : Math.abs(decimals);
			var sep = (typeof thousandsSep === 'undefined') ? ',' : thousandsSep;
			var dec = (typeof decimalSep === 'undefined') ? '.' : decimalSep;
			var s;

			var toFixedFix = function (n, prec) {
				var k = Math.pow(10, prec);
				return '' + (Math.round(n * k) / k).toFixed(prec)
			};

			// @todo: for IE parseFloat(0.55).toFixed(0) = 0;
			s = (prec ? toFixedFix(n, prec) : '' + Math.round(n)).split('.');
			if (s[0].length > 3) {
				s[0] = s[0].replace(/\B(?=(?:\d{3})+(?!\d))/g, sep)
			}
			if ((s[1] || '').length < prec) {
				s[1] = s[1] || '';
				s[1] += new Array(prec - s[1].length + 1).join('0')
			}

			return s.join(dec)
		},

		/**
		 * Empty check similar to PHP.
		 *
		 * @link http://locutus.io/php/empty/
		 * @since 1.2.6
		 */
		empty: function(mixedVar) {

			var undef;
			var key;
			var i;
			var len;
			var emptyValues = [undef, null, false, 0, '', '0'];

			for (i = 0, len = emptyValues.length; i < len; i++) {
				if (mixedVar === emptyValues[i]) {
					return true
				}
			}

			if (typeof mixedVar === 'object') {
				for (key in mixedVar) {
					if (mixedVar.hasOwnProperty(key)) {
						return false;
					}
				}
				return true;
			}

			return false;
		},

		/**
		 * Set cookie container user UUID.
		 *
		 * @since 1.3.3
		 */
		setUserIndentifier: function() {

			if ( ( ( ! window.hasRequiredConsent && typeof wpforms_settings !== 'undefined' && wpforms_settings.uuid_cookie ) || ( window.hasRequiredConsent && window.hasRequiredConsent() ) ) && ! WPForms.getCookie( '_wpfuuid' ) ) {

				// Generate UUID - http://stackoverflow.com/a/873856/1489528
				var s         = new Array(36),
					hexDigits = '0123456789abcdef',
					uuid;

				for (var i = 0; i < 36; i++) {
					s[i] = hexDigits.substr(Math.floor(Math.random() * 0x10), 1);
				}
				s[14] = "4";
				s[19] = hexDigits.substr((s[19] & 0x3) | 0x8, 1);
				s[8]  = s[13] = s[18] = s[23] = '-';

				uuid = s.join("");

				WPForms.createCookie('_wpfuuid', uuid, 3999);
			}
		},

		/**
		 * Create cookie.
		 *
		 * @since 1.3.3
		 */
		createCookie: function(name, value, days) {

			var expires = '';

			// If we have a days value, set it in the expiry of the cookie.
			if ( days ) {
				// If -1 is our value, set a session based cookie instead of a persistent cookie.
				if ( '-1' === days ) {
					expires = '';
				} else {
					var date = new Date();
					date.setTime(date.getTime() + (days*24*60*60*1000));
					expires = '; expires=' + date.toGMTString();
				}
			} else {
				expires = '; expires=Thu, 01 Jan 1970 00:00:01 GMT';
			}

			// Write the cookie.
			document.cookie = name + '=' + value + expires + '; path=/';
		},

		/**
		 * Retrieve cookie.
		 *
		 * @since 1.3.3
		 */
		getCookie: function(name) {

			var nameEQ = name + '=',
				ca     = document.cookie.split(';');

			for ( var i = 0; i < ca.length; i++ ) {
				var c = ca[i];
				while ( c.charAt(0) === ' ' ) {
					c = c.substring(1, c.length);
				}
				if ( c.indexOf(nameEQ) == 0 ) {
					return c.substring(nameEQ.length, c.length);
				}
			}

			return null;
		},

		/**
		 * Delete cookie.
		 */
		removeCookie: function(name) {

			WPForms.createCookie(name, '',-1);
		}
	};

	// Initialize.
	WPForms.init();

	// Add to global scope.
	window.wpforms = WPForms;

})(jQuery);
